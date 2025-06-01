model="models/model-25-05-07T144107FixedRescale-AUGProp-0.0-lr_0.0006_bs_2_tsteps500_sizes128x128x128_gradac_1_numChannels_64_emaStart_100000_updateEMAStep1-milestone-85-step-85000.pt"

export CUDA_VISIBLE_DEVICES=1

python3 ../sample.py --testcsv "../../testScans.csv" --lowDose "jeppes_project/data/reconstructions/OSSART10pct" --fullDose "jeppes_project/data/reconstructions/OSSARTFull"  --input_size 128 --depth_size 128 --num_channels 64 --num_res_blocks 1 --batchsize 1 --weightfile "$model" --timesteps 500

#python3 ../../segment.py --folder "samples/${model}"

# After the job is done, deactivate the virtual environment
deactivate


