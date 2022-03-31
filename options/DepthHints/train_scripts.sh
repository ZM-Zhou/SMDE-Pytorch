# train DepthHins with Stereo in 320x1024 20 epochs
# as the official implementations
CUDA_VISIBLE_DEVICES=7 python\
 train_dist.py\
 --name Depthhints-S_320_bs6\
 --exp_opts options/DepthHints/DepthHints_S_320.yaml\
 --batch_size 6\
 --beta1 0.9\
 --epoch 20\
 --decay_step 5 10 15\
 --decay_rate 0.1\
 --save_freq 5\
 --visual_freq 1000