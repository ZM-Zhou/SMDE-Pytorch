# train FAL-NetB (N=49) with with 192x640 patches
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name FALB-49_192Crop_KITTI_S_B8\
 --exp_opts options/FALB-49/train/fal-net-b49_192crop_kitti_stereo.yaml\
 --batch_size 8\
 --save_freq 10\
 --visual_freq 2000

# train FAL-NetB (N=49) with MOM with with 192x640 patches
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name FALB-49-MOM_192Crop_KITTI_S_B4\
 --exp_opts options/FALB-49/train/fal-net-b49-mom_192crop_kitti_stereo.yaml\
 --batch_size 4\
 --save_freq 10\
 --visual_freq 2000\
 --pretrained_path <.pth>\
 --start_epoch 1\
 --epoch 20

# train FAL-NetB (N=49) with with 192x640 patches
# use both KITTI and Cityscapes
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name FALB-49_192Crop_KITTI-Cityscapes_S_B8\
 --exp_opts options/FALB-49/train/fal-net-b49_192crop_cityscapes-kitti_stereo.yaml\
 --batch_size 8\
 --save_freq 10\
 --visual_freq 2000

# train FAL-NetB (N=49) with MOM with with 192x640 patches
# use both KITTI and Cityscapes
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name FALB-49-MOM_192Crop_KITTI-Cityscapes_S_B4\
 --exp_opts options/FALB-49/train/fal-net-b49-mom_192crop_cityscapes-kitti_stereo.yaml\
 --batch_size 4\
 --save_freq 10\
 --visual_freq 2000\
 --pretrained_path <.pth>\
 --start_epoch 1\
 --epoch 20
