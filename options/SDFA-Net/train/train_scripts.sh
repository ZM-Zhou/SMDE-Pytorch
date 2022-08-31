# train SDFA-Net at stage1
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name SDFA-Net-SwinT-M_192Crop_KITTI_S_St1_B12\
 --exp_opts options/SDFA-Net/train/sdfa_net-swint-m_192crop_kitti_stereo_stage1.yaml\
 --batch_size 12\
 --epoch 25\
 --visual_freq 2000\
 --save_freq 5

# train SDFA-Net at stage2
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name SDFA-Net-SwinT-M_192Crop_KITTI_S_St2_B12\
 --exp_opts options/SDFA-Net/train/sdfa_net-swint-m_192crop_kitti_stereo_stage2.yaml\
 --batch_size 12\
 --visual_freq 2000\
 --save_freq 5\
 --pretrained_path <path to .pth>

# train SDFA-Net at stage1
# on both Cityscapes and KITTI
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name SDFA-Net-SwinT-M_192Crop_Cityscapes-KITTI_S_St1_B12\
 --exp_opts options/SDFA-Net/train/sdfa_net-swint-m_192crop_cityscapes-kitti_stereo_stage1.yaml\
 --batch_size 12\
 --epoch 25\
 --visual_freq 2000\
 --save_freq 5

# train SDFA-Net at stage2
# on both Cityscapes and KITTI
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name SDFA-Net-SwinT-M_192Crop_Cityscapes-KITTI_S_St2_B12\
 --exp_opts options/SDFA-Net/train/sdfa_net-swint-m_192crop_cityscapes-kitti_stereo_stage2.yaml\
 --batch_size 12\
 --visual_freq 2000\
 --save_freq 5\
 --pretrained_path <path to .pth>

