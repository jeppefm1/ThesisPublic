
#export CUDA_VISIBLE_DEVICES=1

# Now you can run your Python script
python3 ../train.py --with_condition --traincsv jeppes_project/Thesis/rigshospitalet/PythonML/trainScans.csv --valcsv jeppes_project/Thesis/rigshospitalet/PythonML/valScans.csv  --lowDose "jeppes_project/data/reconstructions/OSSART10pct" --fullDose "jeppes_project/data/reconstructions/OSSARTFull" --batchsize 2 --gradientAccumSteps 1 --steps 50000 --input_size 128 --depth_size 128 --num_channels 64 --num_res_blocks 1 --timesteps 500 --save_and_sample_every 1000 --train_lr 0.0006 --lrdecaystep 100000 --step_start_ema 100000 --update_ema_every 1 --resume_weight ""  --resume_wandb_id ""
  





