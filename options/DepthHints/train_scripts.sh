# train DepthHins with Stereo in 320x1024 20 epochs
# as the official implementations
CUDA_VISIBLE_DEVICES=2 python\
 train_dist.py\
 --name DepthHints-Res50_320_KITTI_S_B6\
 --exp_opts options/DepthHints/train/depth-hints_res50_320_kitti_stereo.yaml\
 --batch_size 6\
 --beta1 0.9\
 --epoch 20\
 --decay_step 5 10 15\
 --decay_rate 0.1\
 --save_freq 5\
 --visual_freq 1000

 # train DepthHins with Stereo in 192x640 20 epochs
# as the official implementations
CUDA_VISIBLE_DEVICES=3 python\
 train_dist.py\
 --name DepthHints-Res50_192_KITTI_S_B6\
 --exp_opts options/DepthHints/train/depth-hints_res50_192_kitti_stereo.yaml\
 --batch_size 6\
 --beta1 0.9\
 --epoch 20\
 --decay_step 5 10 15\
 --decay_rate 0.1\
 --save_freq 5\
 --visual_freq 1000
