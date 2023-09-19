# train OCFD-Net with 192x640 patches
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name OCFD-Net_192Crop_KITTI_S_B8\
 --exp_opts options/OCFD-Net/train/ocfd-net_192crop_kitti_stereo.yaml\
 --batch_size 8\
 --save_freq 10\
 --visual_freq 2000

# train OCFD-Net with 192x640 patches
# on both kitti and cityscapes dataset
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name OCFD-Net_192Crop_KITTI-Cityscapes_S_B8\
 --exp_opts options/OCFD-Net/train/ocfd-net_192crop_cityscapes-kitti_stereo.yaml\
 --batch_size 8\
 --save_freq 10\
 --visual_freq 2000\
 --num_workers 16
