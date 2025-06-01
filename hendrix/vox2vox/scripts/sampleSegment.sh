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

export CUDA_VISIBLE_DEVICES=0

model="converged_models/25-03-10T124331-AUGProp-0.5-glr_0.0001-dlr_0.0001-bs_2_sizes128x128x128_gradac_1-epoch-1575.pt"

python3 ../sample.py --testcsv "/projects/thesis_zwb495/data/ReconstructedSiemens/test.csv" --targetfolder "/projects/thesis_zwb495/data/ReconstructedSiemens/images"  --input_size 128 --depth_size 128  --weightfile "$model"

python3 ../../misc/segment.py --folder "samples/${model}"

# After the job is done, deactivate the virtual environment
deactivate


