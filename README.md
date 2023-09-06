# Self-supervised Monocular Depth Estimation with Pytorch
The repository is to build a fair environment where the Self-supervised Monocular Depth Estimation (SMDE) methods could be evaluated and developed.
## Welcome to V2.0
In V2.0, you can compute the FLOPs (supported by thop) and infrerence speeds simply. 
We also supports more flexible traning configs such as dividing one training iteration in multiple steps and setting different loss fuctions for different parameters (e.g. used in TiO-Depth). 
We have tried our best to update all the methods in V1.0 to V2.0 and we holp it would be helpful.
BTW, our new method TiO-Depth was accpted to ICCV 2023 !! and it was incloud in this repo.

## About SMDE-Pytorch
We build this repository with Pytorch for evaluating and developing the Self-supervised Monocular Depth Estimation (SMDE) methods. The main targets of the SMDE-Pytorch are:
* Predict depths with typical SMDE methods (with their pretrained models) with simple commands.
* Evaluate the performances (including the FLOPs and speed) of the SMDE methods more fairly.
* Train and modify the existing SMDE methods simply.
* Develop your methods quickly with the modular network parts.

If you have any questions or suggestions, please make an issue or contact us by `zm_zhou1998@163.com` (Maybe I couldn't reply soon due to work.). If you like the work and click the **Star**, we will be happy~

## Setup
We built and tested the repository with Ubuntu 18.04, CUDA 11.0, Python 3.7.9, and Pytorch 1.7.0. For using this repository, we recommend creating a virtual environment by [Anaconda](https://www.anaconda.com/products/individual). Please open a terminal in the root of the repository folder for running the following commands and scripts.
```
conda env create -f environment.yml
conda activate pytorch170cu11
```

## Method Zoo
|Method|Ref.|Test|Train|Paper|Code|
|------|----|----|-----|-----|----|
|[Monodepth2](options/Monodepth2)| 2019 ICCV| ✔| ✔| [Link](https://arxiv.org/abs/1806.01260)| [Link](https://github.com/nianticlabs/monodepth2)|
|[DepthHints](options/DepthHints)| 2019 ICCV| ✔| ✔| [Link](https://arxiv.org/abs/1909.09051)| [Link](https://github.com/nianticlabs/depth-hints)|
|[EdgeOfDepth](options/EdgeOfDepth)| 2020 CVPR| ✔| | [Link](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_The_Edge_of_Depth_Explicit_Constraints_Between_Segmentation_and_Depth_CVPR_2020_paper.html)| [Link](https://github.com/TWJianNuo/EdgeDepth-Release)|
|[PackNet](options/PackNet)| 2020 CVPR| ✔| | [Link](https://openaccess.thecvf.com/content_CVPR_2020/html/Guizilini_3D_Packing_for_Self-Supervised_Monocular_Depth_Estimation_CVPR_2020_paper.html)| [Link](https://github.com/TRI-ML/packnet-sfm)|
|[P2Net](options/P2Net) | 2020 ECCV| ✔| | [Link](https://arxiv.org/abs/2007.07696)| [Link](https://github.com/svip-lab/Indoor-SfMLearner)|
|[FAL-Net](options/FALB-49)| 2020 NeurIPS | ✔| ✔| [Link](https://proceedings.neurips.cc/paper/2020/hash/951124d4a093eeae83d9726a20295498-Abstract.html)| [Link](https://github.com/JuanLuisGonzalez/FAL_net)|
|[HRDepth](options/HRDepth) | 2021 AAAI| ✔| ✔| [Link](https://ojs.aaai.org/index.php/AAAI/article/view/16329)| [Link](https://github.com/shawLyu/HR-Depth)|
|[DIFFNet](options/DIFFNet) | 2021 BMCV| ✔| ✔| [Link](https://arxiv.org/abs/2209.07088) | [Link](https://github.com/ZM-Zhou/SDFA-Net_pytorch)|
|[ManyDepth](options/ManyDepth) | 2021 CVPR| ✔| | [Link](https://arxiv.org/abs/2104.14540)| [Link](https://github.com/nianticlabs/manydepth)|
|[EPCDepth](options/EPCDepth) | 2021 ICCV| ✔| ✔| [Link](https://arxiv.org/abs/2109.12484)| [Link](https://github.com/prstrive/EPCDepth)|
|[FSRE-Depth](options/EPCDepth) | 2021 ICCV| ✔| ✔| [Link](https://openaccess.thecvf.com/content/ICCV2021/html/Jung_Fine-Grained_Semantics-Aware_Representation_Enhancement_for_Self-Supervised_Monocular_Depth_Estimation_ICCV_2021_paper.html)| [Link](https://github.com/hyBlue/FSRE-Depth)|
|[R-MSFM](options/R-MSFM) | 2021 ICCV| ✔| ✔| [Link](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_R-MSFM_Recurrent_Multi-Scale_Feature_Modulation_for_Monocular_Depth_Estimating_ICCV_2021_paper.pdf)| [Link](https://github.com/jsczzzk/R-MSFM)|
|[OCFD-Net](options/OCFD-Net) (Ours)| 2022 ACM-MM'| ✔| ✔| [Link](https://arxiv.org/abs/2203.10925) | [Link](https://github.com/ZM-Zhou/OCFD-Net_pytorch)|
|[SDFA-Net](options/SDFA-Net) (Ours)| 2022 ECCV| ✔| ✔| [Link](https://arxiv.org/abs/2209.07088) | [Link](https://github.com/ZM-Zhou/SDFA-Net_pytorch)|
|[TiO-Depth](options/TiO-Depth) (Ours)| 2023 ICCV| ✔| ✔| [Link](https://arxiv.org/abs/2309.00933) | [Link](https://github.com/ZM-Zhou/TiO-Depth_pytorch)|

* `Test` : You could predict depths with their pretrained models provided by their official implementations. We have tested their performances and more details are given on their pages (click their names in the table).
* `Train`: We have trained the method with this repository and the trained model achieves competitive or better performances compared to the official version.
### TODO List
- [ ] SuperDepth (ICRA 2019)


## Evaluation Results
We give the performances of the methods on **the KITTI raw test set** (an outdoor dataset) for helping you choose the model. More pretrained models are given on their pages (click their names in the above table).
|Method|Info.|Sup|Trained|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|
|------|-----|---|-------|--------|-------|----|-------|--|
|ManyDepth(Mono)|Res18+192x640|Mono|[Offical](https://pan.baidu.com/s/168qFsk68t0117PXwcagtqQ)|0.118|0.891|4.763|0.192|0.871|
|PackNet|PackV1+192x640|Mono|[Official](https://pan.baidu.com/s/1d_uL1q2_bsGEskFDcEfBGA)|0.110|0.836|4.655|0.187|0.881|
|R-MSFM6|Res18+192x640|Mono|[Trained](https://pan.baidu.com/s/1alD5kmgM7P07-hWLDFxyEg)|0.110|0.797|4.646|0.188|0.880|
|Monodepth2|Res18+320x1024|Mono|[Trained](https://pan.baidu.com/s/1T3IGfBB2c5Y2xskACRg3aQ)|0.109|0.797|4.533|0.184|0.888|
|FSRE-Depth|Res18+192x640|Mono|[Trained](https://pan.baidu.com/s/1jFDl9ofeFqxBQGyuGsn1WQ)|0.107|0.751|4.525|0.182|0.886|
|Monodepth2|Res18+320x1024|Stereo|[Trained](https://pan.baidu.com/s/1Kj9HOo15murscIsOchMEUA)|0.104|0.824|4.747|0.200|0.875|
|HRDepth|Res18+384x1280|Mono|[Trained](https://pan.baidu.com/s/1QJhkNhXTRUQimwomRoP96Q)|0.102|0.719|4.396|0.178|0.897|
|FAL-NetB|N=49+375x1242|Stereo|[Trained](https://pan.baidu.com/s/1PhUJ_4s0nm41a49viZRczg)|0.099|0.625|4.197|0.182|0.885|
|DIFFNet|HR18+320x1024|Mono|[Trained](https://pan.baidu.com/s/1xblmyPXNMr_432BN10BUYA)|0.099|0.688|4.345|0.176|0.901|
|DepthHints|Res50+320x1024|Stereo|[Trained](https://pan.baidu.com/s/12xv0IY_hcO1YtsEZJ2Vuog)|0.094|0.680|4.333|0.181|0.894|
|EdgeOfDepth|Res50+320x1024|Stereo|[Official](https://pan.baidu.com/s/1yToYiunNgNQZY8tunZOmGA)|0.092|0.647|4.247|0.177|0.897|
|OCFD-Net|Res50+384x1280|Stereo|[Trained](https://pan.baidu.com/s/1Dep8U4mFnk6czcVqq_qZkA)|0.091|0.576|4.036|0.174|0.901|
|EPCDepth|Res50+320x1024|Stereo|[Trained](https://pan.baidu.com/s/1-Q8N1hPPjKz3BZXbPv_opw)|0.090|0.682|4.282|0.178|0.903|
|SDFA-Net|SwinT*+384x1280|Stereo|[Trained](https://pan.baidu.com/s/1OqnackbFrNYomp_TFYkM5g)|0.089|0.537|3.895|0.169|0.906|
|TiO-Depth|SwinT*+384x1280|Stereo|[Trained](https://pan.baidu.com/s/1rNZvLDcTSGq5XOBFHZRTjg)|0.085|0.544|3.919|0.169|0.911|

The methods on **the NYU v2 test set** (an indoor dataset).
|Method|Info.|Sup|Trained|Abs Rel.|RMSE|log10|A1|
|------|-----|---|-------|--------|----|-----|--|
|P2Net|Res18+5f+288x384|Mono|[Official](https://pan.baidu.com/s/1wpN6O-e453e9n8AJqvG3JA)|0.149|0.556|0.063|0.797|

* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* **code for all the download links is `smde`**

## Predict depth for your image(s) straightforwardly
To predict depth maps for your images, please firstly download the pretrained model that you are interested in from the column named `Trained` in the above table. After unzipping the downloaded model, you could predict the depth maps for your images by
```
python predict.py\
 --image_path <path to your image or folder name for your images>\
 --exp_opts <path to the method training option>\
 --model_path <path to the downloaded or trained model>
```
You also could set `--input_size` to decide the size that the images are reshaped before they are input to the model. If you want to predict on CPU, please set `--cpu`. The depth results `<image name>_pred.npy` and the visualization results `<image name>_visual.png` will be saved in the same folder as the input images.  

For example, if you want to predict depths from the images in `./example_images` with `Monodepth2` (using the model that was saved in `pretrained_models/MD2_S_320_bs4/model/best_model.pth`), you could use:
```
python predict.py\
 --image_path example_images\
 --exp_opts options/Monodepth2/train/monodepth2-res18_320_kitti_stereo.yaml\
 --model_path pretrained_models/MD2_M_320_bs4/model/best_model.pth
```
For the methods which could not be trained in the repository yet, you could use the options in `options/_base/network` for `--exp_opts`. Specifically, you could use the following command for predicting the images with `PackNet` and the pretrained model saved in `pretrained_models/PackNet_M_192_OI/model/PackNet_M_192.pth`.
```
python predict.py\
 --image_path example_images\
 --exp_opts options/_base/networks/packnet.yaml\
 --model_path pretrained_models/PackNet_M_192_OI/model/PackNet_M_192.pth\
```
Since the default image size in `options/_base/networks/packnet.yaml` is `192x640`, when you want to use the model trained under `384x1280`, you could use:
```
python predict.py\
 --image_path example_images\
 --exp_opts options/_base/networks/packnet.yaml\
 --model_path pretrained_models/PackNet_Mv_CS+K_384_OI/model/PackNet_Mv_CS+K_384.pth\
 --input_size 384 1280
```
## Prepare datasets
Before evaluating or training the methods, you should download the used datasets. The datasets that could be used for training or evaluating:
|Dataset|Train|Test|
|-------|-----|----|
|KITTI|✔ (175GB)|✔ (2GB)|
|NYU v2||✔ (2GB)|
|Mak3D||✔ (200MB)|
|Cityscapes|✔ (130GB)|✔ (35GB)|

##### Set data path
We give an example `path_example.py` for setting the path in the repository.
Please create a python file named `path_my.py` and copy the code in `path_example.py` to the `path_my.py`. Then you can replace the used paths to your folder in the `path_my.py`.
the folder for each dataset should be organized like:
```
<root of kitti>
|---2011_09_26
|   |---2011_09_26_drive_0001_sync
|   |   |---image_02
|   |   |---image_03
|   |   |---velodyne_points
|   |   |---...
|   |---2011_09_26_drive_0002_sync
|   |   |---image_02
|   |   |---image_03
|   |   |---velodyne_points
|   |   |---...
|   '''
|---2011_09_28
|   |--- ...
|---gt_depths_raw.npz (for raw Eigen test set)
|---gt_depths_improved.npz (for improved Eigen test set)
```
```
<root of NYU v2 (just test set)>
|---00001.h5
|---00002.h5
|---00003.h5
|---...
```
```
<root of Make3D>
|---Gridlaserdata
|   |---depth_sph_corr-10.21op2-p-015t000.mat
|   |---depth_sph_corr-10.21op2-p-139t000.mat
|   |---...
|---Test134
|   |---img-10.21op2-p-015t000.jpg
|   |---img-10.21op2-p-139t000.jpg
|   |---...
```
```
<root of cityscapes>
|---leftImg8bit
|   |---train
|   |   |---aachen
|   |   |   |---aachen_000000_000019_leftImg8bit.png
|   |   |   |---aachen_000001_000019_leftImg8bit.png
|   |   |   |---...
|   |   |---bochum
|   |   |---...
|   |---train_extra
|   |   |---augsburg
|   |   |---...
|   |---test
|   |   |---...
|   |---val
|   |   |---...
|---rightImg8bit
|   |--- ...
|---camera
|   |--- ...
|---disparity
|   |--- ...
|---gt_depths (for evaluation)
|   |---000_depth.npy
|   |---001_depth.npy
|   |--- ...
```
##### KITTI
For training the methods on the KITTI dataset (the Eigen split), you should download the entire KITTI dataset (about 175GB) by:
```
wget -i ./datasets/kitti_archives_to_download.txt -P <save path>
```
And you could unzip them with:
```
cd <save path>
unzip "*.zip"
```

For evaluating the methods on the KITTI (Eigen raw test set), you should further generate the ground-truth depth file by (as done in the [Monodepth2](https://github.com/nianticlabs/monodepth2)):

```
python datasets/utils/export_kitti_gt_depth.py --data_path <root of KITTI> --split raw
```
If you want to evaluate the method on the KITTI improved test set, you should download the `annotated depth maps` (about 15GB) at [Here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) and unzip it. Then you could generate the imporved ground-truth depth file by:
```
python datasets/utils/export_kitti_gt_depth.py --data_path <root of KITTI> --split improved
```
As an alternative, we provide the Eigen test subset (with `.png` images [Here](https://pan.baidu.com/s/1NejtxajjJt6pQ-VIRJDcUg) or with `.jpg` images [Here](https://pan.baidu.com/s/1AMkcaxh1Ua4cL1VsTXt4Ww), about 2GB) and the generated `gt_depth` files for the people who just want to do the evaluation.

##### NYUv2
We use the NYUv2 test set as done in P2Net and EPCDepth, which could be downloaded in [Here](https://pan.baidu.com/s/1AKv_V59WclGHULt-casXaA)

##### Make3D
We use the Make3D test set for evaluating some methods, which could be downloaded in [Here](http://make3d.cs.cornell.edu/data.html#make3d)

##### Cityscapes
Cityscapes could be used to jointly train the model with KITTI, which is helpful to improve the performance of the model. If you want to use the Cityscapes, please download the following parts of the dataset at [Here](https://www.cityscapes-dataset.com/downloads/) and unzip them to your `<root of cityscapes>` (Note: For some files, you should apply for download permission by email.):
```
leftImg8bit_trainvaltest.zip (11GB)  <- If just do the evluation, download this
leftImg8bit_trainextra.zip (44GB)
rightImg8bit_trainvaltest.zip (11GB)
rightImg8bit_trainextra.zip (44GB)
disparity_trainvaltest.zip (3.5GB)
disparity_trainextra.zip (15GB)
camera_trainvaltest.zip (2MB)  <- If just do the evluation, download this
camera_trainextra.zip (8MB)
```
Then, please generate the camera parameter matrices by:
```
python datasets/utils/export_cityscapes_matrix.py
```
You also need to download the prepared ground-truth depth [Here](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip) which is provided by Watson in ManyDepth.
### Evaluate the methods
To evaluate the methods on the prepared dataset, you could simply use 
```
python evaluate.py\
 --exp_opts <path to the method EVALUATION option>\
 --model_path <path to the downloaded or trained model>
```
We provide the EVALUATION option files in `options/<Method Name>/eval/*`. Here we introduce some important arguments.
|Argument|Information|
|--------|-----------|
|`--metric_name depth_kitti_mono`|Enable the median scaling for the methods traind with monocular sequences (Sup = Mono)|
|`--visual_list`|The samples which you want to save the output (path to a `.txt` file)|
|`--save_pred`|Save the predicted depths of the samples which are in `--visual_list`|
|`--save_visual`|Save the visualization results of the samples which are in `--visual_list`|
|`-fpp`,`-gpp`, `-mspp`|Adopt different post-processing steps. (Please choose one in each time)|

The output files are saved in `eval_res\` by default. Please check `evaluate.py` for more information about arguments.

For example, if you want to evaluate `Monodepth2` on the KITTI Eigen test set with the post-processing proposed by Godard, and you want to save the visualization and predicted depths of all the test samples. Please use:
```
python evaluate.py\
 --exp_opts options/Monodepth2/eval/monodepth2-res18-stereo_320_kitti.yaml\
 --model_path pretrained_models/MD2_S_320_bs4/model/best_model.pth\
 -gpp\
 --save_visual\
 --save_pred\
 --visual_list data_splits/kitti/test_list.txt
```
The evaluation output will be like
```
->Load the test dataset
->Load the pretrained model
->Use the post processing
->Start Evaluation
697/697
    | abs_rel  |  sq_rel  |   rms    | log_rms  |    a1    |    a2    |    a3    |
    |     0.102|     0.795|     4.685|     0.198|     0.876|     0.954|     0.977|
```
The output predicted depths and visualization results will be saved in `eval_res/MD2_S_320_bs4/-gpp/*`.
## Train the methods
To train (reproduce) the methods on the prepared dataset, you could simply use the commands provided in `options/<Method Name>/train/train_scripts.sh`.

For example, if you want to train Monodepth2 on the KITTI dataset with stereo image pairs, please use:
```
python\
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
```

## Modify the methods
coming soon

## References
[Mmsegmentation](https://github.com/open-mmlab/mmsegmentation)  
[Mmcv](https://github.com/open-mmlab/mmcv)  
[Mmengine](https://github.com/open-mmlab/mmengine)  
[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)  
[Monodepth2](https://github.com/nianticlabs/monodepth2)  
[FAL-Net](https://github.com/JuanLuisGonzalez/FAL_net)  
[DepthHints](https://github.com/nianticlabs/depth-hints)  
[DIFFNet](https://github.com/brandleyzhou/DIFFNet)  
[EPCDepth](https://github.com/prstrive/EPCDepth)  
[EdgeOfDepth](https://github.com/TWJianNuo/EdgeDepth-Release)  
[PackNet](https://github.com/TRI-ML/packnet-sfm)  
[P2Net](https://github.com/svip-lab/Indoor-SfMLearner)  
[HRDepth](https://github.com/shawLyu/HR-Depth)  
[FSRE-Depth](https://github.com/hyBlue/FSRE-Depth)  
[ManyDepth](https://github.com/nianticlabs/manydepth)  
[R-MSFM](https://github.com/jsczzzk/R-MSFM)
[ApolloScape Dataset](http://apolloscape.auto/index.html)  
[KITTI Dataset](http://www.cvlibs.net/datasets/kitti/index.php)  
[NYUv2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)  
[Make3D Dataset](http://make3d.cs.cornell.edu/data.html#make3d)  
[Cityscapes Dataset](https://www.cityscapes-dataset.com)
