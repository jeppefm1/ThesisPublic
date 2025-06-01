import torch
from torch.utils.data import DataLoader
import argparse
import datetime
import os
import torchio as tio

from dataset import DicomDataset
from models import GeneratorUNet, Discriminator
from trainer import TrainerGAN


parser = argparse.ArgumentParser()
parser.add_argument('--traincsv', type=str, required=True, help="Path to training CSV file")
parser.add_argument('--valcsv', type=str, default="", help="Path to validation CSV file")
parser.add_argument('--targetfolder', type=str, required=True, help="Path to DICOM dataset folder")
parser.add_argument('--input_size', type=int, default=128, help="Spatial size of input images")
parser.add_argument('--depth_size', type=int, default=128, help="Depth size of input images")
parser.add_argument("--glr", type=float, default=0.0002, help="adam: generator learning rate")
parser.add_argument("--dlr", type=float, default=0.0002, help="adam: discriminator learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--batchsize', type=int, default=1, help="Batch size for training")
parser.add_argument('--epochs', type=int, default=1000) 
parser.add_argument('--val_every', type=int, default=1, help="Validate every N steps")
parser.add_argument('--save_and_sample_every', type=int, default=100)
parser.add_argument('--flipprop', type=float, default=0.2, help="Probability of random flipping in augmentation")
parser.add_argument('--gradientAccumSteps', type=int, default=2)
parser.add_argument('-r', '--resume_weight', type=str, default="")
parser.add_argument('--resume_wandb_id', type=str, default="")
parser.add_argument("--d_threshold", type=float, default=.8, help="discriminator threshold")
parser.add_argument("--weight_decay", type=float, default=2e-4, help="adam: weightdecay")


args = parser.parse_args()


# TorchIO transformation (matching diffusion model)
torchio_transform = tio.Compose([
    tio.RandomFlip(axes=['LR', 'AP', 'IS'], flip_probability=args.flipprop),
])

# Load DicomDataset (matching diffusion model)
train_dataset = DicomDataset(args.traincsv, args.targetfolder, 
                            input_size=args.input_size, 
                            depth_size=args.depth_size, 
                            augment=torchio_transform)

val_dataset = DicomDataset(args.valcsv, args.targetfolder, 
                        input_size=args.input_size, 
                        depth_size=args.depth_size, 
                        augment=None)


# Initialize models
generator = GeneratorUNet().cuda()
discriminator = Discriminator().cuda()

# Model name based on hyperparameters
# Only create a new model name if not resuming
if args.resume_weight == "":
    modelname = f"{datetime.datetime.now().strftime("%y-%m-%dT%H%M%S")}-AUGProp-{args.flipprop}-glr_{args.glr}-dlr_{args.dlr}-bs_{args.batchsize}_sizes{args.input_size}x{args.input_size}x{args.depth_size}_gradac_{args.gradientAccumSteps}"
else:
    modelname = os.path.basename(args.resume_weight).split('-epoch-')[0]


# Initialize trainer
trainer = TrainerGAN(
    generator,
    discriminator,
    train_dataset,
    val_dataset = val_dataset,
    image_size = args.input_size,
    depth_size = args.depth_size,
    train_batch_size = args.batchsize,
    g_lr=args.glr,
    d_lr=args.dlr,
    num_epochs=args.epochs,
    gradient_accumulate_every=args.gradientAccumSteps,
    save_every_n_epochs=args.save_and_sample_every,
    val_every_n_epochs=args.val_every,
    use_wandb=True,
    model_name=modelname,
    lambda_voxel=100,
    d_threshold=args.d_threshold,
    grad_clip_value=1.0,
    betas=(args.b1, args.b2),
    weight_decay = args.weight_decay,
    resume_wandb_id = args.resume_wandb_id

)
if args.resume_weight:
    trainer.load(args.resume_weight)
    print(f"Resumed from checkpoint: {args.resume_weight}")


# Start training
trainer.train()
