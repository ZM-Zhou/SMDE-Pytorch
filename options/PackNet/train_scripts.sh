# train PackNet with Monocular in 192x640 for 50 epochs
# trained in parall mode
CUDA_VISIBLE_DEVICES=0,4 python\
 -m torch.distributed.launch --nproc_per_node=2 --master_port 28066\
 train_dist.py\
 --name Pack-M_192_bs4_parall\
 --exp_opts options/PackNet/Packnet_M_192.yaml\
 --batch_size 2\
 --learning_rate 0.0002\
 --beta1 0.9\
 --epoch 50\
 --decay_step 30\
 --decay_rate 0.5\
 --save_freq 10\
 --visual_freq 2000\
 --metric_name depth_kitti_mono