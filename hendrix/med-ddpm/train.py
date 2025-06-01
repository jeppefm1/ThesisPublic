#Based on https://github.com/mobaidoctor/med-ddpm

from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import DicomDataset
import argparse
import torch
import datetime
import time
import os 
import torchio as tio

parser = argparse.ArgumentParser()
parser.add_argument('--traincsv', type=str, default="")
parser.add_argument('--valcsv', type=str, default="")
parser.add_argument('-t', '--targetfolder', type=str, default="")
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--depth_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--gradientAccumSteps', type=int, default=2)
parser.add_argument('--train_lr', type=float, default=1e-5)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50000) # epochs parameter specifies the number of training iterations
parser.add_argument('--lrdecaystep', type=int, default=2000)
parser.add_argument('--step_start_ema', type=int, default=2000)
parser.add_argument('--update_ema_every', type=int, default=10)
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--flipprop', type=float, default=0.2)
parser.add_argument('--save_and_sample_every', type=int, default=1000)
parser.add_argument('--with_condition', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default="")
parser.add_argument('--resume_wandb_id', type=str, default="")

args = parser.parse_args()

traincsv = args.traincsv
valcsv = args.valcsv
targetfolder = args.targetfolder
input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
resume_weight = args.resume_weight
resume_wandb_id = args.resume_wandb_id
train_lr = args.train_lr
grad_accum_steps = args.gradientAccumSteps
flipprop = args.flipprop
lrdecaystep = args.lrdecaystep
step_start_ema = args.step_start_ema
update_ema_every = args.update_ema_every

torchio_transform = tio.Compose([
    tio.RandomFlip(axes=['LR', 'AP', 'IS'], flip_probability=flipprop),
])


train_dataset = DicomDataset(traincsv, targetfolder, input_size = input_size, depth_size = depth_size, augment=torchio_transform)
val_dataset = DicomDataset(valcsv, targetfolder, input_size = input_size, depth_size = depth_size, augment=None)

in_channels = 2
out_channels = 1

#Create UNET
model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).cuda()

# Only create a new model name if not resuming
if resume_weight == "":
    modelname = f"{datetime.datetime.now().strftime('%y-%m-%dT%H%M%S')}FixedRescale-AUGProp-{flipprop}-lr_{train_lr}_bs_{args.batchsize}_tsteps{args.timesteps}_sizes{input_size}x{input_size}x{depth_size}_gradac_{grad_accum_steps}_numChannels_{num_channels}_emaStart_{step_start_ema}_updateEMAStep{update_ema_every}"
else:
    # Extract modelname from the checkpoint path
    # This assumes the checkpoint filename follows your naming pattern
    modelname = os.path.basename(resume_weight).split('-milestone-')[0].replace('model-', '')
    print(f"Resuming training with model name: {modelname}")


diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l1',    # L1 or L2
    with_condition=with_condition,
    channels=out_channels
).cuda()

if len(resume_weight) > 0:
    weight = torch.load(resume_weight, map_location='cuda')
    diffusion.load_state_dict(weight['ema'])
    print("Model Loaded!")

trainer = Trainer(
    diffusion,
    train_dataset,
    val_dataset = val_dataset,
    image_size = input_size,
    depth_size = depth_size,
    train_batch_size = args.batchsize,
    train_lr = train_lr,
    train_num_steps = args.epochs,         # total training steps
    gradient_accumulate_every = grad_accum_steps,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = True,#True,             # turn on mixed precision training with apex
    with_condition=with_condition,
    save_and_sample_every = save_and_sample_every,
    results_folder = modelname,
    model_name = modelname,
    use_wandb = True,
    resume_wandb_id = resume_wandb_id,
    val_batches = 3,
    val_every=100,
    lr_decay_step=lrdecaystep,
    lr_decay_factor=0.1,
    grad_clip_value=1.0,
    step_start_ema = step_start_ema,
    update_ema_every = update_ema_every
)

# Load weights if resuming
if resume_weight:
    trainer.load(resume_weight)
    print(f"Resumed from checkpoint: {resume_weight}")

trainer.train()
