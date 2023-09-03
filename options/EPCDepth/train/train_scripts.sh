# train EPCDepth with Stereo in 320x1024 20 epochs
# as the official implementations
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name EPCDepth-Res50_320_KITTI_S_b6\
 --exp_opts options/EPCDepth/train/epc-depth_res50_320_kitti_stereo.yaml\
 --batch_size 6\
 --epoch 20\
 --save_freq 5\
 --visual_freq 1000