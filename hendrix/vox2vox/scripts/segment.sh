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

model="converged_models/25-02-24T200254-AUGProp-0.2-glr_0.0002-dlr_0.0002-bs_2_sizes128x128x128_gradac_1-epoch-150.pt"

python3 ../../misc/segment.py --folder "samples/${model}"

# After the job is done, deactivate the virtual environment
deactivate


