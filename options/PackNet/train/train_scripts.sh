# train PackNet with Monocular in 192x640 for 100 epochs
# trained in parall mode
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name PackNet_192_B4_M_Pa\
 --exp_opts options/PackNet/train/packnet_192_kitti_mono.yaml\
 --batch_size 4\
 --learning_rate 0.0002\
 --beta1 0.9\
 --epoch 100\
 --decay_step 60\
 --decay_rate 0.5\
 --save_freq 10\
 --visual_freq 2000\
 --metric_name depth_kitti_mono