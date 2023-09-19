# TiO-Depth
### Abstract
Monocular and binocular self-supervised depth estimations are two important and related tasks in computer vision, which aim to predict scene depths from single images and stereo image pairs respectively. In literature, the two tasks are usually tackled separately by two different kinds of models, and binocular models generally fail to predict depth from single images, while the prediction accuracy of monocular models is generally inferior to binocular models. In this paper, we propose a Two-in-One self-supervised depth estimation network, called TiO-Depth, which could not only compatibly handle the two tasks, but also improve the prediction accuracy. TiO-Depth employs a Siamese architecture and each sub-network of it could be used as a monocular depth estimation model. For binocular depth estimation, a Monocular Feature Matching module is proposed for incorporating the stereo knowledge between the two images, and the full TiO-Depth is used to predict depths. We also design a multi-stage joint-training strategy for improving the performances of TiO-Depth in both two tasks by combining the relative advantages of them. Experimental results on the KITTI, Cityscapes, and DDAD datasets demonstrate that TiO-Depth outperforms both the monocular and binocular state-of-the-art methods in most cases, and further verify the feasibility of a two-in-one network for monocular and binocular depth estimation.

![Image](https://github.com/ZM-Zhou/SMDE-Pytorch/tree/main/options/TiO-Depth/arc.jpg)
### Results
**Results on KITTI raw test set**
|Method|Info.|Train. Data|Sup|PP|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|------|-----|-----------|---|--|--------|-------|----|-------|--|-----|
|TiO-Depth|SwinT*+384x1280|K|Stereo||0.085|0.544|3.919|0.169|0.911|[Baidu](https://pan.baidu.com/s/1rNZvLDcTSGq5XOBFHZRTjg)/[Google](https://drive.google.com/file/d/1XMFA9wQzikjKJ-dZFiPM0pKRr_yB-7E8/view?usp=sharing)|
|TiO-Depth|SwinT*+384x1280|K|Stereo|pp|0.083|0.521|3.864|0.167|0.912|[Baidu](https://pan.baidu.com/s/1rNZvLDcTSGq5XOBFHZRTjg)/[Google](https://drive.google.com/file/d/1XMFA9wQzikjKJ-dZFiPM0pKRr_yB-7E8/view?usp=sharing)|
|TiO-Depth(Bino.)|SwinT*+384x1280|K|Stereo||0.063|0.523|3.611|0.153|0.943|[Baidu](https://pan.baidu.com/s/1rNZvLDcTSGq5XOBFHZRTjg)/[Google](https://drive.google.com/file/d/1XMFA9wQzikjKJ-dZFiPM0pKRr_yB-7E8/view?usp=sharing)|


**Results on KITTI 2015 training set**
|Method|Info.|Train. Data|Sup|PP|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|------|-----|-----------|---|--|--------|-------|----|-------|--|-----|
|TiO-Depth|SwinT*+384x1280|K|Stereo||0.075|0.458|3.717|0.130|0.925|[Baidu](https://pan.baidu.com/s/1NJUA10rLDFJcS2StjbqRrw)/[Google](https://drive.google.com/file/d/1ylElx3LMm70Dmq0t-InwqU60_QcRFAeu/view?usp=sharing)|
|TiO-Depth|SwinT*+384x1280|K|Stereo|pp|0.073|0.439|3.680|0.128|0.925|[Baidu](https://pan.baidu.com/s/1NJUA10rLDFJcS2StjbqRrw)/[Google](https://drive.google.com/file/d/1ylElx3LMm70Dmq0t-InwqU60_QcRFAeu/view?usp=sharing)|
|TiO-Depth(Bino.)|SwinT*+384x1280|K|Stereo||0.050|0.434|3.239|0.104|0.967|[Baidu](https://pan.baidu.com/s/1NJUA10rLDFJcS2StjbqRrw)/[Google](https://drive.google.com/file/d/1ylElx3LMm70Dmq0t-InwqU60_QcRFAeu/view?usp=sharing)|

* code for all the download links of pan Baidu is `smde`.
