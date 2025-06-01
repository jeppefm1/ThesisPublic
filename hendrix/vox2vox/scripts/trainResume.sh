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


python3 ../train.py --traincsv ../../../../data/ReconstructedSiemens/train.csv --valcsv ../../../../data/ReconstructedSiemens/val.csv --targetfolder ../../../../data/ReconstructedSiemens/images --batchsize 2 --gradientAccumSteps 1 --epochs 2000 --input_size 128 --depth_size 128 --save_and_sample_every 2 --resume_weight models/25-03-10T124331-AUGProp-0.5-glr_0.0001-dlr_0.0001-bs_2_sizes128x128x128_gradac_1/25-03-10T124331-AUGProp-0.5-glr_0.0001-dlr_0.0001-bs_2_sizes128x128x128_gradac_1-epoch-831.pt --flipprop 0.5 --val_every 1 --glr 0.0001 --dlr 0.0001 --b1 0.5 --b2 0.999 --d_threshold 0.8 --resume_wandb_id 
  
# After the job is done, deactivate the virtual environment
deactivate






