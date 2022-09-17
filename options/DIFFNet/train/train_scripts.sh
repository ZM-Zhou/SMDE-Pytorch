# train DIFFNet with Monocular in 192x640 for 20 epochs
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name DIFFNet-HR18_192_B16_M\
 --exp_opts options/DIFFNet/train/diffnet_hr18_192_kitti_stereo.yaml\
 --batch_size 16\
 --beta1 0.9\
 --epoch 20\
 --decay_step 15\
 --decay_rate 0.1\
 --save_freq 10\
 --visual_freq 2000\
 --metric_name depth_kitti_mono

# train DIFFNet with Monocular in 192x640 for 20 epochs
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name DIFFNet-HR18_320_B12_M\
 --exp_opts options/DIFFNet/train/diffnet_hr18_320_kitti_stereo.yaml\
 --batch_size 12\
 --beta1 0.9\
 --epoch 20\
 --decay_step 15\
 --decay_rate 0.1\
 --save_freq 10\
 --visual_freq 2000\
 --metric_name depth_kitti_mono