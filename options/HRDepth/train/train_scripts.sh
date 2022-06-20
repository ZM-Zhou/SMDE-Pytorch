# train HRDepth with Monocular in 384x1280 for 20 epochs
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name HRDepth-Res18_384_B12_M\
 --exp_opts options/HRDepth/train/hrdepth-res18_384_kitti_mono.yaml\
 --batch_size 12\
 --beta1 0.9\
 --epoch 20\
 --decay_step 15\
 --decay_rate 0.1\
 --save_freq 10\
 --visual_freq 2000\
 --metric_name depth_kitti_mono