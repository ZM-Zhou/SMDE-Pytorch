# train Monodepth2 with Monocular in 192x640 for 20 epochs
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name MD2-Res50_192_B12_M\
 --exp_opts options/Monodepth2/train/monodepth2-res18_192_kitti_mono.yaml\
 --batch_size 12\
 --beta1 0.9\
 --epoch 20\
 --decay_step 15\
 --decay_rate 0.1\
 --save_freq 10\
 --visual_freq 2000\
 --metric_name depth_kitti_mono

# train Monodepth2 with Stereo in 192x640 for 20 epochs
CUDA_VISIBLE_DEVICES=1 python\
 train_dist.py\
 --name MD2-Res50_192_B12_S\
 --exp_opts options/Monodepth2/train/monodepth2-res18_192_kitti_stereo.yaml\
 --batch_size 12\
 --beta1 0.9\
 --epoch 20\
 --decay_step 15\
 --decay_rate 0.1\
 --save_freq 10\
 --visual_freq 2000

# train Monodepth2 with Monocular in 320x1024 for 5 epochs
# from 192x640 10 epoch model
# lr = 1e-5
CUDA_VISIBLE_DEVICES=6 python\
 train_dist.py\
 --name MD2-Res50_320_B4_M\
 --exp_opts options/Monodepth2/train/monodepth2-res18_320_kitti_mono.yaml\
 --batch_size 4\
 --beta1 0.9\
 --epoch 15\
 --decay_step 10\
 --decay_rate 0.1\
 --save_freq 5\
 --visual_freq 2000\
 --metric_name depth_kitti_mono\
 --pretrained_path train_log/MD2/2022-03-30_17h18m28s_MD2-M_192_bs12/model/last_model10.pth

# train Monodepth2 with Stereo in 320x1024 for 5 epochs
# from 192x640 10 epoch model
# lr = 1e-5
CUDA_VISIBLE_DEVICES=7 python\
 train_dist.py\
 --name MD2-Res50_320_B4_S\
 --exp_opts options/Monodepth2/train/monodepth2-res18_320_kitti_stereo.yaml\
 --batch_size 4\
 --beta1 0.9\
 --epoch 15\
 --decay_step 10\
 --decay_rate 0.1\
 --save_freq 5\
 --visual_freq 2000\
 --pretrained_path train_log/MD2/2022-03-31_02h46m36s_MD2-S_192_bs12/model/last_model10.pth