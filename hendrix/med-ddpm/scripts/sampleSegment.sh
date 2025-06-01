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

model="converged_models/model-25-03-14T165055FixedRescale-AUGProp-0.2-lr_0.0006_bs_2_tsteps500_sizes128x128x128_gradac_1_numChannels_64-milestone-996-step-99600.pt"

export CUDA_VISIBLE_DEVICES=1

python3 ../sample.py --testcsv "/projects/thesis_zwb495/data/ReconstructedSiemens/test.csv" --targetfolder "/projects/thesis_zwb495/data/ReconstructedSiemens/images"  --input_size 128 --depth_size 128 --num_channels 64 --num_res_blocks 1 --batchsize 1 --weightfile "$model" --timesteps 500

python3 ../../misc/segment.py --folder "samples/${model}"

# After the job is done, deactivate the virtual environment
deactivate


