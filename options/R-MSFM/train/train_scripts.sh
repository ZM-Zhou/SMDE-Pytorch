# train R-MSFM6 with Monocular in 192x640 for 40 epochs
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name R-MSFM6_192_M_B12\
 --exp_opts options/R-MSFM/train/r-msfm6_res18_192_kitti_mono.yaml\
 --batch_size 12\
 --epoch 40\
 --save_freq 10\
 --visual_freq 1000\
 --metric_name depth_kitti_mono

# train R-MSFM3 with Monocular in 192x640 for 40 epochs
# trained in parall mode
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name R-MSFM3_192_M_B12\
 --exp_opts options/R-MSFM/train/r-msfm3_res18_192_kitti_mono.yaml\
 --batch_size 12\
 --epoch 40\
 --save_freq 10\
 --visual_freq 1000\
 --metric_name depth_kitti_mono