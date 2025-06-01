export CUDA_VISIBLE_DEVICES=0

model="models/25-05-03T104505-glr_0.0001-dlr_0.0001-bs_2_sizes128x128x128_gradac_1/25-05-03T104505-glr_0.0001-dlr_0.0001-bs_2_sizes128x128x128_gradac_1-epoch-291.pt"

python3 ../sample.py --testcsv "../../testScans.csv" --lowDose "jeppes_project/data/reconstructions/OSSART10pct" --fullDose "jeppes_project/data/reconstructions/OSSARTFull"  --input_size 128 --depth_size 128  --weightfile "$model"

#python3 ../../segment.py --folder "samples/${model}"

# After the job is done, deactivate the virtual environment
deactivate


