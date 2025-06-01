# *Main part of the code is adopted from the following repository: https://github.com/lucidrains/denoising-diffusion-pytorch
#Based on https://github.com/mobaidoctor/med-ddpm
#AI utilized


import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from inspect import isfunction
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
import nibabel as nib
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import wandb
import skimage.metrics
from wandbAutenticate import autenticate

try:
    from apex import amp
    APEX_AVAILABLE = True
    print("APEX: ON")
except:
    APEX_AVAILABLE = False
    print("APEX: OFF")

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn, #UNET MODEL
        *,
        image_size,
        depth_size,
        channels = 1,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None,
        with_condition = False,
        with_pairwised = False,
        apply_bce = False,
        lambda_bce = 0.0
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.depth_size = depth_size
        self.denoise_fn = denoise_fn
        self.with_condition = with_condition
        self.with_pairwised = with_pairwised
        self.apply_bce = apply_bce
        self.lambda_bce = lambda_bce

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef3', to_torch(
            1. - (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t, c=None):
        x_hat = 0
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + x_hat
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, c=None):
        x_hat = 0.
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise -
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat
        )

    def q_posterior(self, x_start, x_t, t, c=None):
        x_hat = 0.
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_hat
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, c = None):
        if self.with_condition:
            #Remove noise
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(torch.cat([x, c], 1), t)) #Call UNET
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t, c=c)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, condition_tensors=None, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, c=condition_tensors, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors = None):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            if self.with_condition:
                t = torch.full((b,), i, device=device, dtype=torch.long)
                img = self.p_sample(img, t, condition_tensors=condition_tensors)
            else:
                img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, batch_size = 2, condition_tensors = None):
        image_size = self.image_size
        depth_size = self.depth_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, depth_size, image_size, image_size), condition_tensors = condition_tensors)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    def q_sample(self, x_start, t, noise=None, c=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_hat = 0.
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise + x_hat
        )

    def p_losses(self, x_start, t, condition_tensors = None, noise = None):
        b, c, h, w, d = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.with_condition:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_fn(torch.cat([x_noisy, condition_tensors], 1), t)
        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, condition_tensors=None, *args, **kwargs):
        b, c, d, h, w, device, img_size, depth_size = *x.shape, x.device, self.image_size, self.depth_size
        assert h == img_size and w == img_size and d == depth_size, f'Expected dimensions: height={img_size}, width={img_size}, depth={depth_size}. Actual: height={h}, width={w}, depth={d}.'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, condition_tensors=condition_tensors, *args, **kwargs)


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        val_dataset=None,
        ema_decay=0.995,
        image_size=128,
        depth_size=128,
        train_batch_size=2,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        fp16=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        val_every=500,
        results_folder='./results',
        with_condition=False,
        use_wandb=False,  # Add flag for wandb
        wandb_project="medDDPMRigshospitalet",  # Wandb project name
        val_batches = 3,
        model_name = "model",
        # Scheduler parameters
        scheduler_patience=5000,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-6,
        scheduler_metric='validation_loss',
        grad_clip_value=1.0,
        resume_wandb_id=None,
        device_ids=[0, 1],  # Specify which GPUs to use
        output_device=0     # Main GPU for outputs 

    ):

        super().__init__()
        self.device_ids = device_ids
        self.output_device = output_device
        
        # Set up model on main device first
        self.model = diffusion_model.cuda(device_ids[0])
        self.ema = EMA(ema_decay)
        # For EMA model, we maintain a separate copy
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model = self.ema_model.cuda(device_ids[0])


        self.update_ema_every = update_ema_every
        self.use_wandb = use_wandb 
        self.resume_wandb_id = resume_wandb_id

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.fp16 = fp16

        self.batch_size = train_batch_size
        self.batch_size_per_gpu = train_batch_size // len(device_ids)
        self.image_size = diffusion_model.image_size
        self.depth_size = depth_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.val_every = val_every
        self.val_batches = val_batches

        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_min_lr = scheduler_min_lr
        self.scheduler_metric = scheduler_metric
        self.initial_lr = train_lr
        self.grad_clip_value = grad_clip_value 

        self.ds = dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, num_workers=3 * len(self.device_ids), pin_memory=True))

        self.val_dataset = val_dataset
        self.val_dl = None
        if val_dataset is not None:
            self.val_dl = cycle(data.DataLoader(val_dataset, batch_size=train_batch_size, shuffle=True, num_workers=3* len(self.device_ids), pin_memory=True))


        self.opt = Adam(self.model.parameters(), lr=train_lr)
        self.scheduler = ReduceLROnPlateau(
            self.opt, 
            mode='min', 
            factor=scheduler_factor, 
            patience=scheduler_patience, 
            min_lr=scheduler_min_lr
        )



        self.train_lr = train_lr
        self.train_batch_size = train_batch_size
        self.with_condition = with_condition
        self.model_name = model_name
        self.step = 0

        # Handle mixed precision setup first if enabled
        if fp16:
            # Initialize amp BEFORE DataParallel
            (self.model, self.ema_model), self.opt = amp.initialize(
                [self.model, self.ema_model], 
                self.opt, 
                opt_level='O1'
            )
        
        #Wrap models with DataParallel (after amp if enabled)
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids, output_device=output_device)
        self.ema_model = torch.nn.DataParallel(self.ema_model, device_ids=device_ids, output_device=output_device)
        
        
        os.makedirs(os.path.join("models", results_folder), exist_ok=True)
        self.log_dir = self.create_log_dir(results_folder)
        results_folder = os.path.join("models",results_folder, "results")
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)#"./logs")
        self.reset_parameters()

        # Initialize wandb if enabled
        if self.use_wandb:
            autenticate()
            
            # Configure wandb initialization
            wandb_init_kwargs = {
                "project": wandb_project,
                "config": {
                    "lr": self.train_lr,
                    "batch_size": self.train_batch_size,
                    "image_size": self.image_size,
                    "depth_size": self.depth_size,
                    "num_steps": self.train_num_steps,
                    "ema_decay": ema_decay,
                    "step_start_ema": step_start_ema,
                    "gradient_accumulate_every": gradient_accumulate_every,
                    "val_batches": self.val_batches,
                    "timesteps": diffusion_model.num_timesteps
                }
            }
            
            # If resuming a run, add the run_id
            if self.resume_wandb_id:
                wandb_init_kwargs["id"] = self.resume_wandb_id
                wandb_init_kwargs["resume"] = "must"
            else:
                wandb_init_kwargs["name"] = model_name
                
            # Initialize wandb with the configured parameters
            wandb.init(**wandb_init_kwargs)
            
            # Save the run ID to the checkpoint for future resuming
            self.wandb_run_id = wandb.run.id

    def to_hu(self, image):
        volume_norm = np.clip(image, -0.99999999999999, 0.99999999999999)
        volume_min = -1000
        volume_max = 2000
        
        # Invert the normalization
        return ((volume_norm + 1) / 2) * (volume_max - volume_min) + volume_min
        


    def validate(self):
        """Compute validation loss, PSNR, SSIM"""
        if self.val_dl is None:
            return  # Skip if no validation set
        
        self.model.eval()
        total_loss, total_psnr, total_ssim = 0, 0, 0
        num_batches = 0

        with torch.no_grad():
            #Number of batches to validate
            for _ in range(self.val_batches): 
                data = next(self.val_dl)
                if self.with_condition:
                    input_tensors = data['input'].cuda()
                    target_tensors = data['target'].cuda()
                    generated = self.ema_model.module.sample(batch_size=input_tensors.shape[0], condition_tensors=input_tensors)
                else:
                    data = data.cuda()
                    generated = self.ema_model.module.sample(batch_size=input_tensors.shape[0], condition_tensors= None)

                loss = F.mse_loss(generated, target_tensors if self.with_condition else data)
                total_loss += loss.item()

                #Compute PSNR & SSIM
                output_np = generated.cpu().numpy()
                target_np = (target_tensors if self.with_condition else data).cpu().numpy()
                

                batch_psnr = []
                batch_ssim = []
                for i in range(output_np.shape[0]):  # Iterate over batch
                    psnr_slices = []
                    ssim_slices = []
                    for z in range(output_np.shape[1]):  # Iterate over depth slices
                        psnr_value = skimage.metrics.peak_signal_noise_ratio(target_np[i, z], output_np[i, z])
                        ssim_value = skimage.metrics.structural_similarity(target_np[i,z], output_np[i,z], multichannel=False, data_range=2.0)

                        psnr_slices.append(psnr_value)
                        ssim_slices.append(ssim_value)

                    batch_psnr.append(np.mean(psnr_slices))  # Average over depth slices
                    batch_ssim.append(np.mean(ssim_slices))

                total_psnr += np.mean(batch_psnr)
                total_ssim += np.mean(batch_ssim)
                num_batches += 1

        avg_val_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        self.model.train()

        self.writer.add_scalar("validation_loss", avg_val_loss, self.step)
        self.writer.add_scalar("PSNR", avg_psnr, self.step)
        self.writer.add_scalar("SSIM", avg_ssim, self.step)

        if self.use_wandb:
            wandb.log({
                "validation_loss": avg_val_loss,
                "PSNR": avg_psnr,
                "SSIM": avg_ssim,
                "step": self.step
            })
        
        return avg_val_loss


    def validate_noise_prediction(self):
        if self.val_dl is None:
            return None 
        
        self.model.eval()
        total_noise_pred_loss = 0
        num_batches = min(2, self.val_batches)
        
        with torch.no_grad():
            for _ in range(num_batches):
                data = next(self.val_dl)
                
                if self.with_condition:
                    input_tensors = data['input'].cuda()
                    target_tensors = data['target'].cuda()
                else:
                    target_tensors = data.cuda()
                    input_tensors = None
                
                # Pick random timesteps for validation
                b = target_tensors.shape[0]
                t = torch.randint(0, self.model.module.num_timesteps, (b,), device=target_tensors.device).long()
                
                # Get noise prediction loss
                if self.with_condition:
                    noise_pred_loss = self.model.module.p_losses(
                        x_start=target_tensors, 
                        t=t, 
                        condition_tensors=input_tensors
                    )
                else:
                    noise_pred_loss = self.model.module.p_losses(
                        x_start=target_tensors, 
                        t=t
                    )
                
                total_noise_pred_loss += noise_pred_loss.item()
        
        avg_noise_pred_loss = total_noise_pred_loss / num_batches
        
        if self.use_wandb:
            wandb.log({
                "val_noise_pred_loss": avg_noise_pred_loss,
                "step": self.step
            })
        
        self.model.train()

        #Check for lr adjustments.
        if self.scheduler_metric == 'validation_loss':
                    self.scheduler.step(avg_noise_pred_loss)

        return avg_noise_pred_loss


    def create_log_dir(self, resultsFolder):
        log_dir = os.path.join("./logs", resultsFolder)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def reset_parameters(self):
       self.ema_model.module.load_state_dict(self.model.module.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model.module, self.model.module)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.module.state_dict(),
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema': self.ema_model.module.state_dict(),
            'wandb_run_id': self.wandb_run_id if self.use_wandb else None 
        }
        torch.save(data, str(self.results_folder / f'model-{self.model_name}-milestone-{milestone}-step-{self.step}.pt'))
        
        # Additionally save latest checkpoint for easy resuming
        torch.save(data, str(self.results_folder / f'model-{self.model_name}-latest.pt'))


    def load(self, checkpoint_path):
        data = torch.load(checkpoint_path)

        self.step = data['step']
        self.model.module.load_state_dict(data['model'])
        self.ema_model.module.load_state_dict(data['ema'])

        # Load scheduler state if available
        if 'scheduler' in data:
            self.scheduler.load_state_dict(data['scheduler'])

        if 'opt' in data:
            self.opt.load_state_dict(data['opt'])

        
        # Set wandb run ID if it exists in the checkpoint
        if 'wandb_run_id' in data and data['wandb_run_id'] is not None:
            self.wandb_run_id = data['wandb_run_id']
            # If we're using wandb but didn't specify a run ID to resume, use the one from the checkpoint
            if self.use_wandb and self.resume_wandb_id is None:
                self.resume_wandb_id = self.wandb_run_id
                print(f"Resuming wandb run with ID: {self.resume_wandb_id}")
                # Re-initialize wandb with the recovered run ID
                wandb.init(id=self.resume_wandb_id, resume="must")
        
        print(f"Loaded checkpoint from step {self.step}")


    def train(self):
        self.writer.add_hparams(
        {
            "lr": self.train_lr,
            "batchsize": self.train_batch_size,
            "image_size": self.image_size,
            "depth_size": self.depth_size
        },
        {}
        )
        self.writer.close()

        backwards = partial(loss_backwards, self.fp16)
        start_time = time.time()

        while self.step < self.train_num_steps:
            accumulated_loss = []
            
            for i in range(self.gradient_accumulate_every):
                # Get data
                if self.with_condition:
                    data = next(self.dl)
                    input_tensors = data['input'].cuda()
                    target_tensors = data['target'].cuda()
                    loss = self.model(target_tensors, condition_tensors=input_tensors)
                else:
                    data = next(self.dl).cuda()
                    loss = self.model(data)
                
                # DataParallel returns a tuple of losses from each GPU, sum them
                if isinstance(loss, tuple):
                    loss = sum(l for l in loss) / len(loss)
                else:
                    loss = loss.sum() / self.batch_size
                
                # Check for infinite/NaN loss
                if not torch.isfinite(loss).all():
                    print(f"Step {self.step}: Loss is infinite or NaN. Skipping batch.")
                    continue
                    
                print(f'{self.step}: {loss.item()}')
                
                # Scale loss
                scaled_loss = loss / self.gradient_accumulate_every
                
                # Backward pass with gradient clipping
                if self.fp16:
                    with amp.scale_loss(scaled_loss, self.opt) as scaled_loss:
                        scaled_loss.backward()
                        # Clip gradients immediately after backward
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.opt), self.grad_clip_value)
                else:
                    scaled_loss.backward()
                    # Clip gradients immediately after backward
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                
                accumulated_loss.append(loss.item())
            #Save loss
            average_loss = np.mean(accumulated_loss)
            end_time = time.time()
            self.writer.add_scalar("training_loss", average_loss, self.step)
            if self.use_wandb:
                wandb.log({
                    "training_loss": average_loss,
                    "step": self.step,
                    "learning_rate": self.opt.param_groups[0]['lr'],
                    "time_elapsed (min)": (end_time - start_time) / 60
                })

            if self.val_dataset is not None:
                noise_pred_loss = self.validate_noise_prediction()
                print(f'{self.step}: Validation noise prediction loss: {noise_pred_loss:.6f}')


            self.opt.step()
            self.opt.zero_grad()

            #Step ema
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            #Validate
            if self.step % self.val_every == 0 and self.val_dataset is not None:
                self.validate()

            #Save and sample
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                
                if self.val_dl is not None:
                    self.model.eval()
                    with torch.no_grad():
                        #Take one image from validation set
                        val_data = next(self.val_dl)
                        val_data = {key: value[:1] for key, value in val_data.items()}
                        input_tensors = val_data['input'].cuda() if self.with_condition else val_data.cuda()
                        target_tensors = val_data['target'].cuda() if self.with_condition else val_data.cuda()
                        
                        generated = self.ema_model.module.sample(batch_size=input_tensors.shape[0], condition_tensors=input_tensors if self.with_condition else None)

                    sampleImage = generated.cpu().numpy().reshape([self.image_size, self.image_size, self.depth_size])
                    huImg = self.to_hu(sampleImage)
                    nifti_img = nib.Nifti1Image(huImg, affine=np.eye(4))
                    nib.save(nifti_img, str(self.results_folder / f'sample-{milestone}-step-{self.step}.nii.gz'))
                    if self.use_wandb:
                        # Create a combined visualization with middle slices
                        middle_slice_idx = self.depth_size // 2
                        
                        # Take middle slices
                        input_slice = input_tensors.cpu().numpy().reshape([self.image_size, self.image_size, self.depth_size])[:, :, middle_slice_idx]
                        generated_slice = sampleImage[:, :, middle_slice_idx]
                        target_slice = target_tensors.cpu().numpy().reshape([self.image_size, self.image_size, self.depth_size])[:, :, middle_slice_idx]
                        
                        
                        # Create a combined image (horizontally stacked)
                        combined_slice = np.concatenate([input_slice, generated_slice, target_slice], axis=1)
                        
                        # Log both individual and combined images
                        wandb.log({
                            "combined_view": wandb.Image(combined_slice, caption=f"Milestone {milestone} - Input | Generated | Target"),
                            "input_image": wandb.Image(input_slice, caption=f"Input - Milestone {milestone}"),
                            "generated_image": wandb.Image(generated_slice, caption=f"Generated - Milestone {milestone}"),
                            "target_image": wandb.Image(target_slice, caption=f"Target - Milestone {milestone}")
                        })

                #Save model
                self.save(milestone)

            self.step += 1

        print('training completed')
        milestone = self.step+1 // self.save_and_sample_every
        self.save(milestone)
        end_time = time.time()
        execution_time = (end_time - start_time)/3600
        if self.use_wandb:
            wandb.log({"execution_time (hour)": execution_time, "last_loss": average_loss})
            wandb.finish()
