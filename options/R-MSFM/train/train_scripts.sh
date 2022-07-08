# train R-MSFM6 with Monocular in 192x640 for 40 epochs
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name R-MSFM6_192_M_B12\
 --exp_opts options/R-MSFM/train/r-msfm6_res18_192_kitti_mono.yaml\
 --batch_size 12\
 --learning_rate 0.0002\
 --optim_name AdamW
 --weight_decay 0.00005\
 --beta1 0.9\
 --clip_grad 1\
 --epoch 40\
 --decay_step 40\
 --decay_rate 0.1\
 --save_freq 10\
 --visual_freq 1000\
 --metric_name depth_kitti_mono

# train R-MSFM3 with Monocular in 192x640 for 40 epochs
# trained in parall mode
CUDA_VISIBLE_DEVICES0 python\
 train_dist.py\
 --name R-MSFM3_192_M_B12\
 --exp_opts options/R-MSFM/train/r-msfm3_res18_192_kitti_mono.yaml\
 --batch_size 12\
 --learning_rate 0.0002\
 --optim_name AdamW\
 --weight_decay 0.00005\
 --beta1 0.9\
 --clip_grad 1\
 --epoch 40\
 --decay_step 40\
 --decay_rate 0.1\
 --save_freq 10\
 --visual_freq 1000\
 --metric_name depth_kitti_mono
