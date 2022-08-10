# train FSRE-Depth with Monocular in 192x640 for 20 epochs
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name FSREDepth-Res18_192_KITTI_M_B12\
 --exp_opts options/FSRE-Depth/train/fsre-depth-res18_192_kitti_mono.yaml\
 --batch_size 12\
 -lr 0.00015\
 --beta1 0.9\
 --epoch 20\
 --decay_step 10 15\
 --decay_rate 0.1\
 --save_freq 10\
 --visual_freq 2000\
 --metric_name depth_kitti_mono
