# OCFD-Net
### Abstract
Self-supervised monocular depth estimation, aiming to learn scene depths from single images in a self-supervised manner, has received much attention recently. In spite of recent efforts in this field, how to learn accurate scene depths and alleviate the negative influence of occlusions for self-supervised depth estimation, still remains an open problem. Addressing this problem, we firstly empirically analyze the effects of both the continuous and discrete depth constraints which are widely used in the training process of many existing works. Then inspired by the above empirical analysis, we propose a novel network to learn an Occlusion-aware Coarse-to-Fine Depth map for self-supervised monocular depth estimation, called OCFD-Net. Given an arbitrary training set of stereo image pairs, the proposed OCFD-Net does not only employ a discrete depth constraint for learning a coarse-level depth map, but also employ a continuous depth constraint for learning a scene depth residual, resulting in a fine-level depth map. In addition, an occlusion-aware module is designed under the proposed OCFD-Net, which is able to improve the capability of the learnt fine-level depth map for handling occlusions. Experimental results on KITTI demonstrate that the proposed method outperforms the comparative state-of-the-art methods under seven commonly used metrics in most cases. In addition, experimental results on Make3D demonstrate the effectiveness of the proposed method in terms of the cross-dataset generalization ability under four commonly used metrics.

![Image](https://github.com/ZM-Zhou/SMDE-Pytorch/tree/main/options/OCFD-Net/OCFD-Arc.jpg)
### Results
**Results on KITTI raw test set**
|Method|Info.|Train. Data|Sup|PP|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|------|-----|-----------|---|--|--------|-------|----|-------|--|-----|
|OCFD-Net|Res50+384x1280|K|Stereo||0.091|0.576|4.036|0.174|0.901|[Baidu](https://pan.baidu.com/s/1Dep8U4mFnk6czcVqq_qZkA)|
|OCFD-Net|Res50+384x1280|K|Stereo|pp|0.090|0.563|4.005|0.172|0.903|[Baidu](https://pan.baidu.com/s/1Dep8U4mFnk6czcVqq_qZkA)|
|OCFD-Net|Res50+384x1280|CS+K|Stereo||0.088|0.554|3.944|0.171|0.906|[Baidu](https://pan.baidu.com/s/1m76zppemVS1PnNnlgyYPCQ)|
|OCFD-Net|Res50+384x1280|CS+K|Stereo|pp|0.086|0.536|3.889|0.169|0.909|[Baidu](https://pan.baidu.com/s/1m76zppemVS1PnNnlgyYPCQ)|


**Results on KITTI improved test set**
|Method|Info.|Train. Data|Sup|PP|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|------|-----|-----------|---|--|--------|-------|----|-------|--|-----|
|OCFD-Net|Res50+384x1280|K|Stereo||0.070|0.270|2.821|0.104|0.949|[Baidu](https://pan.baidu.com/s/1Dep8U4mFnk6czcVqq_qZkA)|
|OCFD-Net|Res50+384x1280|K|Stereo|pp|0.069|0.262|2.785|0.103|0.951|[Baidu](https://pan.baidu.com/s/1Dep8U4mFnk6czcVqq_qZkA)|
|OCFD-Net|Res50+384x1280|CS+K|Stereo||0.068|0.246|2.669|0.099|0.955|[Baidu](https://pan.baidu.com/s/1m76zppemVS1PnNnlgyYPCQ)|
|OCFD-Net|Res50+384x1280|CS+K|Stereo|pp|0.066|0.236|2.612|0.096|0.957|[Baidu](https://pan.baidu.com/s/1m76zppemVS1PnNnlgyYPCQ)|


**Results on Make3D test set** (trained on KITTI)
|Method|Info.|Train. Data|Sup|PP|Abs Rel.|Sq Rel.|RMSE|log10|Model|
|------|-----|-----------|---|--|--------|-------|----|-----|-----|
|OCFD-Net|Res50+256x512|K|Stereo||0.279|2.573|6.421|0.145|[Baidu](https://pan.baidu.com/s/1Dep8U4mFnk6czcVqq_qZkA)|
|OCFD-Net|Res50+256x512|K|Stereo|pp|0.275|2.515|6.354|0.144|[Baidu](https://pan.baidu.com/s/1Dep8U4mFnk6czcVqq_qZkA)|
|OCFD-Net|Res50+256x512|CS+K|Stereo||0.259|2.258|5.939|0.137|[Baidu](https://pan.baidu.com/s/1m76zppemVS1PnNnlgyYPCQ)|
|OCFD-Net|Res50+256x512|CS+K|Stereo|pp|0.256|2.187|5.856|0.135|[Baidu](https://pan.baidu.com/s/1m76zppemVS1PnNnlgyYPCQ)|


* code for all the download links is `smde`
