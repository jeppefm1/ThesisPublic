#AI utilized
#Based on https://github.com/enochkan/vox2vox/tree/master
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import wandb
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pathlib import Path
import nibabel as nib
from torch.autograd import Variable
from tqdm import tqdm
import skimage.metrics
from wandbAutenticate import autenticate
import warnings
warnings.filterwarnings(
    "ignore",
    message="Using TorchIO images without a torchio.SubjectsLoader",
    module="torchio.data.image"
)
torch.backends.cudnn.enabled = False

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

class TrainerGAN:
    def __init__(
        self,
        generator,
        discriminator,
        dataset,
        val_dataset=None,
        image_size=128,
        depth_size=128,
        train_batch_size=2,
        g_lr=1e-4,
        d_lr=1e-4,
        num_epochs=100,
        gradient_accumulate_every=1,
        save_every_n_epochs=1,
        val_every_n_epochs=1,
        use_wandb=False,
        wandb_project="Vox2VoxRigshospitalet",
        model_name="gan_model",
        lambda_voxel=100,
        d_threshold=0.8,
        grad_clip_value=1.0,
        betas=(0.5, 0.999),
        weight_decay = 2e-4,
        resume_wandb_id=None,
        device_ids = [1,0], # Prioritize GPU 1 first, then 0
        max_valsamples = 30,
        # Scheduler parameters
        scheduler_patience=50,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-6,
        scheduler_metric='validation_loss'
    ):
        super().__init__()

        self.device_ids = device_ids 
        print(f"Using GAN device priority: {self.device_ids}")
        
        # Set priority GPU for CUDA operations
        torch.cuda.set_device(self.device_ids[0])

        # Initialize models - first corrected the initialization order
        self.generator = generator
        self.discriminator = discriminator
        
        # Move models to priority GPU
        self.generator = self.generator.cuda(self.device_ids[0])
        self.discriminator = self.discriminator.cuda(self.device_ids[0])
        
        # Set up data parallelism
        if len(self.device_ids) > 1:
            print(f"GAN using data parallelism across GPUs: {self.device_ids}")
            self.generator = torch.nn.DataParallel(self.generator, device_ids=self.device_ids)
            self.discriminator = torch.nn.DataParallel(self.discriminator, device_ids=self.device_ids)

        
        self.use_wandb = use_wandb
        self.resume_wandb_id = resume_wandb_id

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.depth_size = depth_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.num_epochs = num_epochs
        self.save_every_n_epochs = save_every_n_epochs
        self.val_every_n_epochs = val_every_n_epochs
        self.grad_clip_value = grad_clip_value
        self.lambda_voxel = lambda_voxel
        self.d_threshold = d_threshold
        self.weight_decay = weight_decay
        self.max_valsamples = max_valsamples

        # Scheduler parameters
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_min_lr = scheduler_min_lr
        self.scheduler_metric = scheduler_metric


        # Initialize dataloaders
        self.ds = dataset
        self.dl = DataLoader(
            self.ds, 
            batch_size=train_batch_size, 
            shuffle=True, 
            num_workers=3, 
            pin_memory=True
        )

        self.val_dataset = val_dataset
        self.val_dl = None
        if val_dataset is not None:
            self.val_dl = DataLoader(
                val_dataset, 
                batch_size=1, 
                shuffle=True, 
                num_workers=3, 
                pin_memory=True
            )

        # Calculate output of image discriminator (PatchGAN)
        self.patch = (1, image_size // 2**4, image_size // 2**4, depth_size // 2**4)
        
        # Loss functions
        self.criterion_GAN = torch.nn.MSELoss().cuda()
        self.criterion_voxelwise = torch.nn.L1Loss().cuda()

        # Optimizers
        self.opt_G = Adam(generator.parameters(), lr=g_lr, betas=betas, weight_decay=weight_decay)
        self.opt_D = Adam(discriminator.parameters(), lr=d_lr, betas=betas, weight_decay=weight_decay)

        # Schedulers
        self.scheduler_G = ReduceLROnPlateau(
            self.opt_G, 
            mode='min', 
            factor=scheduler_factor, 
            patience=scheduler_patience, 
            min_lr=scheduler_min_lr
        )
        self.scheduler_D = ReduceLROnPlateau(
            self.opt_D, 
            mode='min', 
            factor=scheduler_factor, 
            patience=scheduler_patience, 
            min_lr=scheduler_min_lr
        )

        self.model_name = model_name
        self.current_epoch = 0
        
        # Setup directories and logging
        self.results_folder = Path(os.path.join("models", model_name), exist_ok=True)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = self.create_log_dir(model_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        if self.use_wandb:
            autenticate()

            wandb_init_kwargs = {
                "project" : wandb_project ,
                "config": {
                    "g_lr": g_lr,
                    "d_lr": d_lr,
                    "batch_size": train_batch_size,
                    "image_size": image_size,
                    "depth_size": depth_size,
                    "num_epochs": num_epochs,
                    "gradient_accumulate_every" : gradient_accumulate_every,
                    "grad_clip_value" : grad_clip_value,
                    "lambda_voxel" : lambda_voxel,
                    "d_threshold" : d_threshold,
                    "weight_decay" : weight_decay,
                    "scheduler_patience": scheduler_patience,
                    "scheduler_factor": scheduler_factor,
                    "scheduler_min_lr": scheduler_min_lr,
                    "scheduler_metric": scheduler_metric
            }
            }   

            if self.resume_wandb_id:
                wandb_init_kwargs["id"] = self.resume_wandb_id
                wandb_init_kwargs["resume"] = "must"
            else:
                wandb_init_kwargs["name"] = model_name

            wandb.init(**wandb_init_kwargs)
            wandb.define_metric("validation_loss", summary="min")
            wandb.define_metric("PSNR", summary="max")
            wandb.define_metric("SSIM", summary="max")

            self.wandb_run_id = wandb.run.id

    def create_log_dir(self, results_folder):
        log_dir = os.path.join("./logs", results_folder)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def save(self, epoch):
        generator_state = self.generator.module.state_dict() if isinstance(self.generator, torch.nn.DataParallel) else self.generator.state_dict()
        discriminator_state = self.discriminator.module.state_dict() if isinstance(self.discriminator, torch.nn.DataParallel) else self.discriminator.state_dict()
        
        data = {
            'epoch': epoch,
            'generator': generator_state,
            'discriminator': discriminator_state,
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'scheduler_G': self.scheduler_G.state_dict(),
            'scheduler_D': self.scheduler_D.state_dict(),
            'wandb_run_id': self.wandb_run_id if self.use_wandb else None
        }
        torch.save(data, str(self.results_folder / f'{self.model_name}-epoch-{epoch}.pt'))

    def load(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        if isinstance(self.generator, torch.nn.DataParallel):
            self.generator.module.load_state_dict(checkpoint['generator'])
            self.discriminator.module.load_state_dict(checkpoint['discriminator'])
        else:
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
        
        # Load optimizer state
        self.opt_G.load_state_dict(checkpoint['opt_G'])
        self.opt_D.load_state_dict(checkpoint['opt_D'])
        
        # Load scheduler state if available
        if 'scheduler_G' in checkpoint and 'scheduler_D' in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G'])
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D'])
        
        # Set current epoch
        self.current_epoch = checkpoint['epoch']
        
        # Get wandb run ID if it exists
        if self.use_wandb and 'wandb_run_id' in checkpoint:
            self.wandb_run_id = checkpoint['wandb_run_id']
            self.resume_wandb_id = self.wandb_run_id
            wandb.init(id=self.resume_wandb_id, resume="must")
            
        print(f"Resuming from epoch {self.current_epoch}")

    
    def to_hu(self, image):
        volume_norm = np.clip(image, -0.99999999999999, 0.99999999999999)
        volume_min = -1000
        volume_max = 2000
        
        # Invert the normalization
        return ((volume_norm + 1) / 2) * (volume_max - volume_min) + volume_min

    def train_one_epoch(self):
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_losses = []
        epoch_d_losses = []
        epoch_d_accuracies = []
        epoch_gan_losses = []
        epoch_voxel_losses = []
        
        for batch_idx, batch in tqdm(enumerate(self.dl)):
            real_A = Variable(batch["input"].cuda())
            real_B = Variable(batch["target"].cuda())
        
            # Ground truths
            valid = Variable(torch.ones((real_A.size(0), *self.patch)).cuda())
            fake = Variable(torch.zeros((real_A.size(0), *self.patch)).cuda())
            
            # Train Discriminator
            self.opt_D.zero_grad()
            
            # Generate fake sample
            fake_B = self.generator(real_A)
            
            # Real loss
            pred_real = self.discriminator(real_B, real_A)
            loss_real = self.criterion_GAN(pred_real, valid)
            
            # Fake loss
            pred_fake = self.discriminator(fake_B.detach(), real_A)
            loss_fake = self.criterion_GAN(pred_fake, fake)
            
            # Total discriminator loss
            loss_D = 0.5 * (loss_real + loss_fake)
            
            # Calculate discriminator accuracy
            d_real_acc = torch.ge(pred_real.squeeze(), 0.5).float()
            d_fake_acc = torch.le(pred_fake.squeeze(), 0.5).float()
            d_total_acc = torch.mean(torch.cat((d_real_acc, d_fake_acc), 0))
            
            # Only update discriminator if accuracy is below threshold
            if d_total_acc <= self.d_threshold:
                loss_D.backward()
                self.opt_D.step()
            
            # Train Generator
            self.opt_G.zero_grad()
            
            # Generate fake sample
            fake_B = self.generator(real_A)
            pred_fake = self.discriminator(fake_B, real_A)
            
            # Generator losses
            loss_GAN = self.criterion_GAN(pred_fake, valid)
            loss_voxel = self.criterion_voxelwise(fake_B, real_B)
            
            # Total generator loss
            loss_G = loss_GAN + self.lambda_voxel * loss_voxel
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip_value)
            self.opt_G.step()
            
            # Store batch losses
            epoch_g_losses.append(loss_G.item())
            epoch_d_losses.append(loss_D.item())
            epoch_d_accuracies.append(d_total_acc.item())
            epoch_gan_losses.append(loss_GAN.item())
            epoch_voxel_losses.append(loss_voxel.item())
            
        # Return average losses for the epoch
        return {
            'g_loss': np.mean(epoch_g_losses),
            'd_loss': np.mean(epoch_d_losses),
            'd_accuracy': np.mean(epoch_d_accuracies),
            'gan_loss': np.mean(epoch_gan_losses),
            'voxel_loss': np.mean(epoch_voxel_losses),
        }

    def train(self):
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            print(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            
            # Train for one epoch
            epoch_stats = self.train_one_epoch()
            
            # Logging
            self.writer.add_scalar("epoch_loss_G", epoch_stats['g_loss'], epoch)
            self.writer.add_scalar("epoch_loss_D", epoch_stats['d_loss'], epoch)
            self.writer.add_scalar("epoch_d_accuracy", epoch_stats['d_accuracy'], epoch)

            current_lr_G = self.opt_G.param_groups[0]['lr']
            current_lr_D = self.opt_D.param_groups[0]['lr']
            
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "epoch_loss_G": epoch_stats['g_loss'],
                    "epoch_loss_D": epoch_stats['d_loss'],
                    "epoch_d_accuracy": epoch_stats['d_accuracy'],
                    "epoch_gan_loss": epoch_stats['gan_loss'],
                    "epoch_voxel_loss": epoch_stats['voxel_loss'],
                    "learning_rate_G": current_lr_G,
                    "learning_rate_D": current_lr_D
                })
            
            # Save model and generate samples
            if epoch % self.save_every_n_epochs == 0:
                self.save(epoch + 1)
                with torch.no_grad():
                    sample_batch = next(iter(self.val_dl))
                    sample_A = Variable(sample_batch["input"].cuda())
                    sample_B = Variable(sample_batch["target"].cuda())
                    affine =  Variable(sample_batch['new_affine'].numpy())
                    header =  Variable(sample_batch['new_header'])
                    fake_sample = self.generator(sample_A)
                    
                    # Move tensors to CPU and convert to numpy
                    input_image = sample_A.cpu().numpy()
                    generated_image = fake_sample.cpu().numpy()
                    target_image = sample_B.cpu().numpy()
                    
                    # Remove batch dimension if batch_size=1
                    if input_image.shape[0] == 1:
                        input_image = np.squeeze(input_image, axis=0)
                        generated_image = np.squeeze(generated_image, axis=0)
                        target_image = np.squeeze(target_image, axis=0)
                        
                    # Remove channel dimension if channels=1
                    if len(input_image.shape) > 3 and input_image.shape[0] == 1:
                        input_image = np.squeeze(input_image, axis=0)
                        generated_image = np.squeeze(generated_image, axis=0)
                        target_image = np.squeeze(target_image, axis=0)
                    
                    # Save generated image as NIFTI
                    huImg = self.to_hu(generated_image)
                    nifti_img = nib.Nifti1Image(huImg, affine=affine, header=header)
                    nib.save(nifti_img, str(self.results_folder / f'sample-epoch-{epoch}.nii.gz'))
                    
                    if self.use_wandb:
                        # Create a combined visualization with middle slices
                        middle_slice_idx = self.depth_size // 2
                        
                        # Take middle slices
                        input_slice = input_image[:, :, middle_slice_idx]
                        generated_slice = generated_image[:, :, middle_slice_idx]
                        target_slice = target_image[:, :, middle_slice_idx]
                        
                        
                        # Create a combined image (horizontally stacked)
                        combined_slice = np.concatenate([input_slice, generated_slice, target_slice], axis=1)
                        
                        # Log both individual and combined images
                        wandb.log({
                            "combined_view": wandb.Image(combined_slice, caption=f"Epoch {epoch} - Input | Generated | Target"),
                            "input_image": wandb.Image(input_slice, caption=f"Input - Epoch {epoch}"),
                            "generated_image": wandb.Image(generated_slice, caption=f"Generated - Epoch {epoch}"),
                            "target_image": wandb.Image(target_slice, caption=f"Target - Epoch {epoch}")
                        })
            
            # Validation if available
            if self.val_dl is not None and (epoch + 1) % self.val_every_n_epochs == 0:
                self.generator.eval()
                val_g_losses = []
                val_g_psnr = []
                val_g_ssim = []
                samples_processed = 0
                
                # Use torch.no_grad() for validation to save memory
                with torch.no_grad():
                    for val_batch in self.val_dl:
                        val_real_A = val_batch["input"].cuda(self.device_ids[0])
                        val_real_B = val_batch["target"].cuda(self.device_ids[0])
                        batch_size = val_real_A.shape[0]
                        
                        # Check if we've processed enough samples
                        if self.max_valsamples is not None and samples_processed >= self.max_valsamples:
                            break
                            
                        # If this batch would exceed our max_valsamples, only take what we need
                        if self.max_valsamples is not None and samples_processed + batch_size > self.max_valsamples:
                            # Calculate how many more samples we need
                            samples_needed = self.max_valsamples - samples_processed
                            val_real_A = val_real_A[:samples_needed]
                            val_real_B = val_real_B[:samples_needed]
                            
                        val_fake_B = self.generator(val_real_A)
                        val_loss_voxel = self.criterion_voxelwise(val_fake_B, val_real_B)
                        val_g_losses.append(val_loss_voxel.item())
                        
                        # Move tensors to CPU and convert to numpy once before the loop
                        real_B_np = val_real_B.cpu().numpy()
                        fake_B_np = val_fake_B.cpu().numpy()
                        
                        # Compute PSNR and SSIM for each batch
                        batch_size = val_fake_B.shape[0]
                        batch_psnr = np.zeros(batch_size)
                        batch_ssim = np.zeros(batch_size)
                        
                        for i in range(batch_size):
                            # Calculate metrics across all slices for each sample in batch
                            slices_count = val_fake_B.shape[1]
                            sample_psnr = np.zeros(slices_count)
                            sample_ssim = np.zeros(slices_count)
                            
                            for z in range(slices_count):
                                sample_psnr[z] = skimage.metrics.peak_signal_noise_ratio(
                                    real_B_np[i, z], fake_B_np[i, z]
                                )
                                sample_ssim[z] = skimage.metrics.structural_similarity(
                                    real_B_np[i, z], fake_B_np[i, z], 
                                    multichannel=False, data_range=2.0
                                )
                            
                            batch_psnr[i] = np.mean(sample_psnr)
                            batch_ssim[i] = np.mean(sample_ssim)
                        
                        val_g_psnr.append(np.mean(batch_psnr))
                        val_g_ssim.append(np.mean(batch_ssim))
                        samples_processed += batch_size
                
                # Calculate average metrics
                avg_val_loss = np.mean(val_g_losses)
                avg_val_psnr = np.mean(val_g_psnr)
                avg_val_ssim = np.mean(val_g_ssim)

                self.writer.add_scalar("val_loss", avg_val_loss, epoch)
                self.writer.add_scalar("val_psnr", avg_val_psnr, epoch)
                self.writer.add_scalar("val_ssim", avg_val_ssim, epoch)
                if self.use_wandb:
                    wandb.log({
                        "validation_loss": avg_val_loss,
                        "PSNR": avg_val_psnr,
                        "SSIM": avg_val_ssim,
                        "epoch": epoch
                    })

            # Step the schedulers based on validation metrics
                if self.scheduler_metric == 'validation_loss':
                    self.scheduler_G.step(avg_val_loss)
                    self.scheduler_D.step(avg_val_loss)
                elif self.scheduler_metric == 'SSIM':
                    # For SSIM, higher is better, so we use negative value for the scheduler
                    self.scheduler_G.step(-avg_val_ssim)
                    self.scheduler_D.step(-avg_val_ssim)
                else:
                    # Default to training loss if the metric is not specified
                    self.scheduler_G.step(epoch_stats['g_loss'])
                    self.scheduler_D.step(epoch_stats['d_loss'])
        
        print('Training completed.')
        execution_time = (time.time() - start_time) / 3600
        self.writer.add_hparams(
            {"execution_time_hours": execution_time},
            {
                "final_loss_G": epoch_stats['g_loss'],
                "final_loss_D": epoch_stats['d_loss']
            }
        )
        self.writer.close()
        
        if self.use_wandb:
            wandb.log({
                "execution_time_hours": execution_time,
                "final_loss_G": epoch_stats['g_loss'],
                "final_loss_D": epoch_stats['d_loss']
            })
            wandb.finish()