# train EPCDepth with Stereo in 320x1024 20 epochs
# as the official implementations
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name EPCDepth-Res50_320_KITTI_S_b6\
 --exp_opts options/EPCDepth/train/epc-depth_res50_320_kitti_stereo.yaml\
 --batch_size 6\
 --beta1 0.9\
 --epoch 20\
 --decay_step 15\
 --decay_rate 0.1\
 --save_freq 5\
 --visual_freq 1000