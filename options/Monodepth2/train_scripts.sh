# train Monodepth2 with Monocular in 192x640 for 20 epochs
CUDA_VISIBLE_DEVICES=6 python\
 train_dist.py\
 --name MD2-M_192_bs12\
 --exp_opts options/Monodepth2/MD2_M_192.yaml\
 --batch_size 12\
 --beta1 0.9\
 --epoch 20\
 --decay_step 15\
 --decay_rate 0.1\
 --save_freq 10\
 --visual_freq 2000\
 --metric_name depth_kitti_mono

# train Monodepth2 with Stereo in 192x640 for 20 epochs
CUDA_VISIBLE_DEVICES=7 python\
 train_dist.py\
 --name MD2-S_192_bs12\
 --exp_opts options/Monodepth2/MD2_S_192.yaml\
 --batch_size 12\
 --beta1 0.9\
 --epoch 20\
 --decay_step 15\
 --decay_rate 0.1\
 --save_freq 10\
 --visual_freq 2000

# train Monodepth2 with Stereo in 320x1024 for 5 epochs
# from 192x640 10 epoch model
CUDA_VISIBLE_DEVICES=7 python\
 train_dist.py\
 --name MD2-S_320_bs4\
 --exp_opts options/Monodepth2/MD2_S_320.yaml\
 --batch_size 4\
 --beta1 0.9\
 --epoch 15\
 --decay_step 15\
 --decay_rate 0.1\
 --save_freq 5\
 --visual_freq 2000\
 --pretrained_path 