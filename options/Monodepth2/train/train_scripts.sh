# train Monodepth2 with Monocular in 192x640 for 20 epochs
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name MD2-Res18_192_B12_M\
 --exp_opts options/Monodepth2/train/monodepth2-res18_192_kitti_mono.yaml\
 --batch_size 12\
 --epoch 20\
 --save_freq 10\
 --visual_freq 2000\
 --metric_name depth_kitti_mono

# train Monodepth2 with Stereo in 192x640 for 20 epochs
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name MD2-Res18_192_B12_S\
 --exp_opts options/Monodepth2/train/monodepth2-res18_192_kitti_stereo.yaml\
 --batch_size 12\
 --epoch 20\
 --save_freq 10\
 --visual_freq 2000

# train Monodepth2 with Monocular in 320x1024 for 5 epochs
# from 192x640 10 epoch model
# lr = 1e-5
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name MD2-Res18_320_B4_M\
 --exp_opts options/Monodepth2/train/monodepth2-res18_320_kitti_mono.yaml\
 --batch_size 4\
 --epoch 20\
 --save_freq 10\
 --visual_freq 2000\
 --metric_name depth_kitti_mono
 --pretrained_path <*/last_model10.pth>

# train Monodepth2 with Stereo in 320x1024 for 5 epochs
# from 192x640 10 epoch model
# lr = 1e-5
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name MD2-Res18_320_B4_S\
 --exp_opts options/Monodepth2/train/monodepth2-res18_320_kitti_stereo.yaml\
 --batch_size 4\
 --epoch 20\
 --save_freq 10\
 --visual_freq 2000\
 --pretrained_path <*/last_model10.pth>