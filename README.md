# Self-supervised Monocular Depth Estimation with Pytorch
The repository is to build a fair environment where the Self-supervised Monocular Depth Estimation (SMDE) methods could be evaluated and developed.
## About SMDE-Pytorch
We build this repository with Pytorch for evaluating and developing the Self-supervised Monocular Depth Estimation (SMDE) methods. The main targets of the SMDE-Pytorch are:
* Predict depths with typical SMDE methods (with their pretrained models) by simple commands.
* Evaluate the performances of the SMDE methods more fairly.
* Train and modify the existing SMDE methods simply (coming soon).
* Develop your methods quickly with the modular network parts (coming soon).

If you have any questions or suggestions, please make an issue or contact us by `zhouzhengming@ia.ac.cn`. If you like the work and click the **Star**, we will be happy~

## Setup
We built and tested the repository with Ubuntu 18.04, CUDA 11.0, Python 3.7.9, and Pytorch 1.7.0. For using this repository, we recommend creating a virtual environment by [Anaconda](https://www.anaconda.com/products/individual). Please open a terminal in the root of the repository folder for running the following commands and scripts.
```
conda env create -f environment.yml
conda activate pytorch170cu11
```

## Method Zoo
|Method|Test|Train|Paper|Code|
|------|----|-----|-----|----|
|[Monodepth2](options/Monodepth2)| ✔| ✔| [Link](https://arxiv.org/abs/1806.01260)| [Link](https://github.com/nianticlabs/monodepth2)|
|[FAL-Net](options/FALB-49) | ✔| ✔ | [Link](https://proceedings.neurips.cc/paper/2020/hash/951124d4a093eeae83d9726a20295498-Abstract.html)| [Link](https://github.com/JuanLuisGonzalez/FAL_net)|
|[DepthHints](options/DepthHints) | ✔| | [Link](https://arxiv.org/abs/1909.09051)| [Link](https://github.com/nianticlabs/depth-hints)|
|[EPCDepth](options/EPCDepth) | ✔| | [Link](https://arxiv.org/abs/2109.12484)| [Link](https://github.com/prstrive/EPCDepth)|


* `Test` : You could predict depths with their pretrained models provided by their official implementations. We have tested their performances and more details are given on their pages (click their names in the table).
* `Train`: We have trained the method with this repository and the trained model achieves competitive or better performances compared to the official version.
### TODO List
- [x] Monodepth2 (ICCV 2019)
- [x] FAL-Net (NeurIPS 2020)
- [x] DepthHints (ICCV 2019)
- [X] EPCDepth (ICCV 2021)
- [ ] Edge-of-depth (CVPR 2020)
- [ ] PackNet (CVPR 2020)
- [ ] Check the post-process

## Evaluation Results
We give the performances of the methods on **the KITTI raw test set** (an outdoor dataset) for helping you choose the model. More pretrained models are given on their pages (click their names in the above table).
|Method|Info.|Sup|Trained|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|
|------|-----|---|-----|--------|-------|----|-------|--|
|Monodepth2|Res18+192x640|Mono|[Trained](https://pan.baidu.com/s/154ib4uD1Gp-ly4OSKHyTNw)|0.112|0.856|4.774|0.190|0.880|
|Monodepth2|Res18+320x1024|Stereo|[Trained](https://pan.baidu.com/s/1Je1yhuYoa25eTUbS57kj4A)|0.106|0.798|4.700|0.202|0.871|
|FAL-NetB|N=49+375x1242|Stereo|[Trained](https://pan.baidu.com/s/1PhUJ_4s0nm41a49viZRczg)|0.099|0.625|4.197|0.182|0.885|
|DepthHints|Res50+320x1024|Stereo|[OI](https://pan.baidu.com/s/1OPesveOI0us8rVEwal-pGg)|0.097|0.737|4.448|0.186|0.889|
|EPCDepth|Res50+320x1024|Stereo|[OI](https://pan.baidu.com/s/1X4TWog23u2Wk6m6H_mbApA)|0.096|0.684|4.278|0.184|0.889|

* `OI` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* **code for all the download links is `smde`**

## Predict for your image(s)
To predict depth maps for your images, please firstly download the pretrained model that you are interested in from the column named `Trained` in the above table. After unzipping the downloaded model, you could predict the depth maps for your images by
```
python predict.py\
 --image_path <path to your image or folder name for your images>\
 --exp_opts <path to the method option>\
 --model_path <path to the downloaded or trained model>
```
You could set `--input_size` to decide the size that the images are reshped before they are input to the model. If you want to predict on CPU, please set `--cpu`. The depth resutls `<image name>_pred.npy` and the visualization resutls `<image name>_visual.png` will be saved in the same folder as the input images.  
For example, if you want to predict depths from the images in `./example_images` with `Monodepth2` (the model was saved in `pretrained_models/MD2_S_320_bs4/model/best_model.pth`), you could use:
```
python predict.py\
 --image_path example_images\
 --exp_opts options/Monodepth2/MD2_S_320.yaml\
 --model_path pretrained_models/MD2_S_320_bs4/model/best_model.pth
```

## Evaluate the method
coming soon

## Train the method
coming soon

## References
[Mmsegmentation](https://github.com/open-mmlab/mmsegmentation)  
[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)  
[Monodepth2](https://github.com/nianticlabs/monodepth2)  
[FAL-Net](https://github.com/JuanLuisGonzalez/FAL_net)  
[DepthHints](https://github.com/nianticlabs/depth-hints)  
[EPCDepth](https://github.com/prstrive/EPCDepth)  
[ApolloScape Dataset](http://apolloscape.auto/index.html)  
[KITTI Dataset](http://www.cvlibs.net/datasets/kitti/index.php)  