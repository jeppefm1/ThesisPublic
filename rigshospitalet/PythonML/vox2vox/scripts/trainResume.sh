
#export CUDA_VISIBLE_DEVICES=1
python3 ../train.py --traincsv ../../trainScans.csv --valcsv ../../trainScans.csv --lowDose jeppes_project/data/reconstructions/OSSART10pct --fullDose jeppes_project/data/reconstructions/OSSARTFull --batchsize 2 --gradientAccumSteps 1 --epochs 1000 --input_size 128 --depth_size 128 --save_and_sample_every 5 --resume_weight jeppes_project/Thesis/rigshospitalet/PythonML/vox2vox/scripts/models/25-05-03T104505-glr_0.0001-dlr_0.0001-bs_2_sizes128x128x128_gradac_1/25-05-03T104505-glr_0.0001-dlr_0.0001-bs_2_sizes128x128x128_gradac_1-epoch-196.pt --val_every 1 --glr 0.00005 --dlr 0.00005 --b1 0.5 --b2 0.999 --d_threshold 0.8 --resume_wandb_id 513zn291 



