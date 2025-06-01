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

model="converged_models/model-25-02-20T145501FixedRescale-AUGProp-0.2-lr_0.0001_bs_2_tsteps500_sizes128x128x128_gradac_1_numChannels_64-milestone-4-step-400.pt"

python3 ../../misc/segment.py --folder "samples/${model}"

# After the job is done, deactivate the virtual environment
deactivate


