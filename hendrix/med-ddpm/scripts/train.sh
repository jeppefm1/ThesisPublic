#!/bin/bash
# SLURM options
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32GB
#SBATCH --time=1-23:59:59
#SBATCH -p gpu --gres=gpu:a40:1
 
# load module
module load python cuda

source ../../misc/env/bin/activate

#export CUDA_VISIBLE_DEVICES=1

# Now you can run your Python script
python3 ../train.py --with_condition --traincsv ../../../../data/ReconstructedSiemens/train.csv --valcsv ../../../../data/ReconstructedSiemens/val.csv --targetfolder ../../../../data/ReconstructedSiemens/images --batchsize 2 --gradientAccumSteps 1 --epochs 60000 --input_size 128 --depth_size 128 --num_channels 64 --num_res_blocks 1 --timesteps 500 --save_and_sample_every 250 --train_lr 1e-5 --lrdecaystep 100000 --step_start_ema 100000 --update_ema_every 1 --resume_weight ""  --resume_wandb_id ""
  
# After the job is done, deactivate the virtual environment
deactivate






