# train FAL-NetB (N=49) with with 192x640 patches
CUDA_VISIBLE_DEVICES=2 python\
 train_dist.py\
 --name FALB-49_192Crop_KITTI_S_B8\
 --exp_opts options/FALB-49/train/fal-net-b49_192crop_kitti_stereo.yaml\
 --batch_size 8\
 --save_freq 10\
 --visual_freq 2000

# train FAL-NetB (N=49) with with 192x640 patches
CUDA_VISIBLE_DEVICES=3 python\
 train_dist.py\
 --name FALB-49-MOM_192Crop_KITTI_S_B4\
 --exp_opts options/FALB-49/train/fal-net-b49-mom_192crop_kitti_stereo.yaml\
 --batch_size 4\
 --save_freq 10\
 --visual_freq 2000\
 --pretrained_path <.pth>\
 -lr 0.00005\
 --decay_step 10\
 --start_epoch 1\
 --epoch 20
