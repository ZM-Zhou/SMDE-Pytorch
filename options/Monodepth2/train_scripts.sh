CUDA_VISIBLE_DEVICES=1 python\
 train_dist.py\
 --name MD2-S_192_bs12\
 --exp_opts options/Monodepth2/MD2_S_192.yaml\
 --batch_size 12\
 --beta1 0.9\
 --epoch 20\
 --decay_step 15\
 --decay_rate 0.1\
 --save_freq 5\
 --visual_freq 2000
 