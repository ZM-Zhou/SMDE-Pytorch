# train TiO-Depth with Stereo in 256x832 for 50 epochs
# trained with KITTI
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name TiO-Depth-Swint-M_rc256_KITTI_S_B8\
 --exp_opts options/TiO-Depth/train/tio_depth-swint-m_384crop256_kitti_stereo.yaml\
 --batch_size 8\
 --metric_source rawdepth sdepth refdepth\
 --save_freq 5\
 --visual_freq 1000

# train TiO-Depth with Stereo in 256x832 for 50 epochs
# trained with KITTI Full
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name TiO-Depth-Swint-M_rc256_KITTIfull_S_B8\
 --exp_opts options/TiO-Depth/train/tio_depth-swint-m_384crop256_kittifull_stereo.yaml\
 --batch_size 8\
 --metric_source rawdepth sdepth refdepth\
 --metric_name depth_kitti_stereo2015\
 --save_freq 5\
 --visual_freq 1000
