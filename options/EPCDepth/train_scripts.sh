# train EPCDepth with Stereo in 320x1024 20 epochs
# the train protocol is same with DepthHints
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name EPCDepth-S_320_bs3\
 --exp_opts options/EPCDepth/EPCDepth-Res50_320.yaml\
 --batch_size 3\
 --beta1 0.9\
 --epoch 20\
 --decay_step 5 10 15\
 --decay_rate 0.1\
 --save_freq 5\
 --visual_freq 1000

# train EPCDepth with Stereo in 320x1024 20 epochs
# the train protocol is same with Official Imp.
# trained in parall mode
CUDA_VISIBLE_DEVICES=0,1 python\
 -m torch.distributed.launch --nproc_per_node=2 --master_port 28067\
 train_dist.py\
 --name EPCDepth-Res50_320_KITTI_S_b6_parall\
 --exp_opts options/EPCDepth/train/epc-depth_res50_320_kitti_stereo.yaml\
 --batch_size 3\
 --beta1 0.9\
 --epoch 20\
 --decay_step 15\
 --decay_rate 0.1\
 --save_freq 5\
 --visual_freq 1000