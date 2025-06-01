#!/bin/bash
# SLURM options
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32GB
#SBATCH --time=1-23:59:59
#SBATCH -p gpu --gres=gpu:l40s:1
 
# load module
module load python cuda

source ../../misc/env/bin/activate

export CUDA_VISIBLE_DEVICES=1

# Now you can run your Python script
python3 ../train.py --with_condition --traincsv ../../../../data/ReconstructedSiemens/train.csv --valcsv ../../../../data/ReconstructedSiemens/val.csv --targetfolder ../../../../data/ReconstructedSiemens/images --batchsize 2 --gradientAccumSteps 1 --epochs 100000 --input_size 128 --depth_size 128 --num_channels 64 --num_res_blocks 1 --timesteps 500 --save_and_sample_every 250 --train_lr 5e-5 --lrdecaystep 200000 --step_start_ema 2000 --update_ema_every 1 --resume_weight models/25-03-20T151235FixedRescale-AUGProp-0.2-lr_5e-05_bs_2_tsteps500_sizes128x128x128_gradac_1_numChannels_64_emaStart_2000_updateEMAStep1/results/model-25-03-20T151235FixedRescale-AUGProp-0.2-lr_5e-05_bs_2_tsteps500_sizes128x128x128_gradac_1_numChannels_64_emaStart_2000_updateEMAStep1-milestone-60000-step-60000.pt --resume_wandb_id 
  
# After the job is done, deactivate the virtual environment
deactivate






