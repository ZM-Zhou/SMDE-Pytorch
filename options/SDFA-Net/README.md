# SDFA-Net
### Abstract
Self-supervised monocular depth estimation has received much attention recently in computer vision. Most of the existing works in literature aggregate multi-scale features for depth prediction via either straightforward concatenation or element-wise addition, however, such feature aggregation operations generally neglect the contextual consistency between multi-scale features. Addressing this problem, we propose the Self-Distilled Feature Aggregation (SDFA) module for simultaneously aggregating a pair of low-scale and high-scale features and maintaining their contextual consistency. The SDFA employs three branches to learn three feature offset maps respectively: one offset map for refining the input low-scale feature and the other two for refining the input highscale feature under a designed self-distillation manner. Then, we propose an SDFA-based network for self-supervised monocular depth estimation, and design a self-distilled training strategy to train the proposed network with the SDFA module. Experimental results on the KITTI dataset demonstrate that the proposed method outperforms the comparative state-of-the-art methods in most cases.

![Image](https://github.com/ZM-Zhou/SMDE-Pytorch/tree/main/options/SDFA-Net/arch-v2.jpg)
### Results
**Results on KITTI raw test set**
|Method|Info.|Train. Data|Sup|PP|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|------|-----|-----------|---|--|--------|-------|----|-------|--|-----|
|SDFA-Net|SwinT*+384x1280|K|Stereo||0.089|0.537|3.895|0.169|0.906|[Baidu](https://pan.baidu.com/s/1OqnackbFrNYomp_TFYkM5g)/[Google](https://drive.google.com/file/d/1RxCJ6lz6MpeHIPLNFmm1hJikeDUOBXu8/view?usp=sharing)|
|SDFA-Net|SwinT*+384x1280|K|Stereo|pp|0.088|0.530|3.864|0.168|0.907|[Baidu](https://pan.baidu.com/s/1OqnackbFrNYomp_TFYkM5g)/[Google](https://drive.google.com/file/d/1RxCJ6lz6MpeHIPLNFmm1hJikeDUOBXu8/view?usp=sharing)|
|SDFA-Net|SwinT*+384x1280|CS+K|Stereo||0.084|0.528|3.887|0.167|0.911|[Baidu](https://pan.baidu.com/s/1sHgR5YvIUxye5XVjhRBZOg)/[Google](https://drive.google.com/file/d/11QJJ1WEQ8Z80JUz7zCmq9t9LBMVvzqYD/view?usp=sharing)|
|SDFA-Net|SwinT*+384x1280|CS+K|Stereo|pp|0.084|0.521|3.832|0.166|0.913|[Baidu](https://pan.baidu.com/s/1sHgR5YvIUxye5XVjhRBZOg)/[Google](https://drive.google.com/file/d/11QJJ1WEQ8Z80JUz7zCmq9t9LBMVvzqYD/view?usp=sharing)|


**Results on KITTI improved test set**
|Method|Info.|Train. Data|Sup|PP|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|------|-----|-----------|---|--|--------|-------|----|-------|--|-----|
|SDFA-Net|SwinT*+384x1280|K|Stereo||0.073|0.231|2.581|0.101|0.955|[Baidu](https://pan.baidu.com/s/1OqnackbFrNYomp_TFYkM5g)/[Google](https://drive.google.com/file/d/1RxCJ6lz6MpeHIPLNFmm1hJikeDUOBXu8/view?usp=sharing)|
|SDFA-Net|SwinT*+384x1280|K|Stereo|pp|0.073|0.227|2.545|0.101|0.956|[Baidu](https://pan.baidu.com/s/1OqnackbFrNYomp_TFYkM5g)/[Google](https://drive.google.com/file/d/1RxCJ6lz6MpeHIPLNFmm1hJikeDUOBXu8/view?usp=sharing)|
|SDFA-Net|SwinT*+384x1280|CS+K|Stereo||0.069|0.214|2.542|0.096|0.962|[Baidu](https://pan.baidu.com/s/1sHgR5YvIUxye5XVjhRBZOg)/[Google](https://drive.google.com/file/d/11QJJ1WEQ8Z80JUz7zCmq9t9LBMVvzqYD/view?usp=sharing)|
|SDFA-Net|SwinT*+384x1280|CS+K|Stereo|pp|0.068|0.206|2.471|0.095|0.963|[Baidu](https://pan.baidu.com/s/1sHgR5YvIUxye5XVjhRBZOg)/[Google](https://drive.google.com/file/d/11QJJ1WEQ8Z80JUz7zCmq9t9LBMVvzqYD/view?usp=sharing)|

* code for all the download links of pan Baidu is `smde`.
