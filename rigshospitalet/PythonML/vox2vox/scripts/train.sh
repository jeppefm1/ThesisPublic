#export CUDA_VISIBLE_DEVICES=1
python3 ../train.py --traincsv "../../trainScans.csv" --valcsv "../../trainScans.csv" --lowDose "jeppes_project/data/reconstructions/OSSART10pct" --fullDose "jeppes_project/data/reconstructions/OSSARTFull" --batchsize 2 --gradientAccumSteps 3 --epochs 10000 --input_size 128 --depth_size 128  --save_and_sample_every 5  --resume_weight ""  --val_every 1 --glr 0.005 --dlr 0.0001 --b1 0.5 --b2 0.999 --d_threshold 0.8
  







