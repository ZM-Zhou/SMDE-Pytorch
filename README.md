# Self-supervised Monocular Depth Estimation with Pytorch
The repository is to build a fair environment where the Self-supervised Monocular Depth Estimation (SMDE) methods could be evaluated and developed.
## About SMDE-Pytorch
We build this repository with Pytorch for evaluating and developing the Self-supervised Monocular Depth Estimation (SMDE) methods. The main targets of the SMDE-Pytorch are:
* Predict depths with typical SMDE methods (with their pretrained models) by simple commands.
* Evaluate the performances of the SMDE methods more fairly.
* Train and modify the existing SMDE methods simply (coming soon).
* Develop your methods quickly with the modular network parts (coming soon).

If you have any questions or suggestions, please make an issue or contact us by `zhouzhengming2020@ia.ac.cn`. If you like the work and click the **Star**, we will be happy~

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
|[FAL-Net](options/FALB-49)| 2020 NeurIPS | ✔| ✔| [Link](https://proceedings.neurips.cc/paper/2020/hash/951124d4a093eeae83d9726a20295498-Abstract.html)| [Link](https://github.com/JuanLuisGonzalez/FAL_net)|
|[P2Net](options/P2Net) | 2020 ECCV| ✔| | [Link](https://arxiv.org/abs/2007.07696)| [Link](https://github.com/svip-lab/Indoor-SfMLearner)|
|[EdgeOfDepth](options/EdgeOfDepth)| 2020 CVPR| ✔| | [Link](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_The_Edge_of_Depth_Explicit_Constraints_Between_Segmentation_and_Depth_CVPR_2020_paper.html)| [Link](https://github.com/TWJianNuo/EdgeDepth-Release)|
|[PackNet](options/PackNet)| 2020 CVPR| ✔| | [Link](https://openaccess.thecvf.com/content_CVPR_2020/html/Guizilini_3D_Packing_for_Self-Supervised_Monocular_Depth_Estimation_CVPR_2020_paper.html)| [Link](https://github.com/TRI-ML/packnet-sfm)|
|[EPCDepth](options/EPCDepth) | 2021 ICCV| ✔| ✔| [Link](https://arxiv.org/abs/2109.12484)| [Link](https://github.com/prstrive/EPCDepth)|


* `Test` : You could predict depths with their pretrained models provided by their official implementations. We have tested their performances and more details are given on their pages (click their names in the table).
* `Train`: We have trained the method with this repository and the trained model achieves competitive or better performances compared to the official version.
### TODO List
- [x] Monodepth2 (ICCV 2019)
- [x] DepthHints (ICCV 2019)
- [x] FAL-Net (NeurIPS 2020)
- [x] P^2-Net (ECCV 2020)
- [x] Edge-of-depth (CVPR 2020)
- [x] PackNet (CVPR 2020)
- [X] EPCDepth (ICCV 2021)
- [x] Check the post-process

## Evaluation Results
We give the performances of the methods on **the KITTI raw test set** (an outdoor dataset) for helping you choose the model. More pretrained models are given on their pages (click their names in the above table).
|Method|Info.|Sup|Trained|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|
|------|-----|---|-------|--------|-------|----|-------|--|
|PackNet|PackV1+192x640|Mono|[Official](https://pan.baidu.com/s/1d_uL1q2_bsGEskFDcEfBGA)|0.110|0.836|4.655|0.187|0.881|
|Monodepth2|Res18+320x1024|Mono|[Trained](https://pan.baidu.com/s/1T3IGfBB2c5Y2xskACRg3aQ)|0.109|0.797|4.533|0.184|0.888|
|Monodepth2|Res18+320x1024|Stereo|[Trained](https://pan.baidu.com/s/1Kj9HOo15murscIsOchMEUA)|0.104|0.824|4.747|0.200|0.875|
|FAL-NetB|N=49+375x1242|Stereo|[Trained](https://pan.baidu.com/s/1PhUJ_4s0nm41a49viZRczg)|0.099|0.625|4.197|0.182|0.885|
|DepthHints|Res50+320x1024|Stereo|[Trained](https://pan.baidu.com/s/12xv0IY_hcO1YtsEZJ2Vuog)|0.094|0.680|4.333|0.181|0.894|
|EdgeOfDepth|Res50+320x1024|Stereo|[Official](https://pan.baidu.com/s/1yToYiunNgNQZY8tunZOmGA)|0.092|0.647|4.247|0.177|0.897|
|EPCDepth|Res50+320x1024|Stereo|[Trained](https://pan.baidu.com/s/1-Q8N1hPPjKz3BZXbPv_opw)|0.090|0.682|4.282|0.178|0.903|

The methods on **the NYU v2 test set** (an indoor dataset).
|Method|Info.|Sup|Trained|Abs Rel.|RMSE|log10|A1|
|------|-----|---|-------|--------|----|-----|--|
|P2Net|Res18+5f+288x384|Mono|[Official](https://pan.baidu.com/s/1wpN6O-e453e9n8AJqvG3JA)|0.149|0.556|0.063|0.797|

* `Official` means that the results are predicted with the models got from their Official Implementations.
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
### Prepare data
Before evaluate the performances of the methods, you should download the used datasets and set the path as the **Create data path** step.
##### KITTI Eigen test set
For evaluate the methods on the KITTI Eigen test set, you could download the entire KITTI dataset (about 175GB) by:
```
wget -i ./datasets/kitti_archives_to_download.txt -P <save path>
```
Then, you can unzip them with:
```
cd <save path>
unzip "*.zip"
```
Then you should generate the `gt_depth` file as done in the [Monodepth2](https://github.com/nianticlabs/monodepth2).

As an alternative, we provide the Eigen test subset (with `.png` images [Here](https://pan.baidu.com/s/16qfBtfHp61d8EOFQFv-OWw) or with `.jpg` images [Here](https://pan.baidu.com/s/17w77UwXecqJf8gV3a26hDQ), about 2GB) and the `gt_depth` files for the people who just want to do the evaluation. You can download and unzip it to your data folder.

##### NYUv2 test set
We use the NYUv2 test set as done in P2Net and EPCDepth, which could be downloaded in [Here](https://pan.baidu.com/s/1AKv_V59WclGHULt-casXaA)

##### Set data path
We give an example `path_example.py` for setting the path in the repository.
Please create a python file named `path_my.py` and copy the content in `path_example.py` to the `path_my.py`. Then you can replace the used paths to your folder in the `path_my.py`.

### Do evaluate
To evaluate the method on the prepared dataset, you could simply use 
```
python evaluate.py\
 --exp_opts <path to the method EVALUATION option>\
 --model_path <path to the downloaded or trained model>
```
We provide the EVALUATION option files for the KITTI Eigen test set in `options/<Method Name>/eval/*`, and we give the evaluation commands for each method in its page . 
Here we introduce some important arguments.
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
 --exp_opts options/Monodepth2/eval/MD2_S_320_eval.yaml\
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
## Train the method
coming soon

## References
[Mmsegmentation](https://github.com/open-mmlab/mmsegmentation)  
[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)  
[Monodepth2](https://github.com/nianticlabs/monodepth2)  
[FAL-Net](https://github.com/JuanLuisGonzalez/FAL_net)  
[DepthHints](https://github.com/nianticlabs/depth-hints)  
[EPCDepth](https://github.com/prstrive/EPCDepth)  
[EdgeOfDepth](https://github.com/TWJianNuo/EdgeDepth-Release)  
[PackNet](https://github.com/TRI-ML/packnet-sfm)  
[P2Net](https://github.com/svip-lab/Indoor-SfMLearner)  
[ApolloScape Dataset](http://apolloscape.auto/index.html)  
[KITTI Dataset](http://www.cvlibs.net/datasets/kitti/index.php)  
[NYUv2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
