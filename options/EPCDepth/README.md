# EPCDepth
**Results on KITTI raw test set**
|Backbone|Resolution|Sup|PP|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|--------|----------|---|--|-----|--------|-------|----|-------|--|-----|
|Res50|320x1024|Stereo|gpp|Reported|0.091|0.646|4.207|0.176|0.901|-|
|Res50|320x1024|Stereo|gpp|Official|0.095|0.667|4.239|0.183|0.889|[Baidu](https://pan.baidu.com/s/1X4TWog23u2Wk6m6H_mbApA)|
|Res50|320x1024|Stereo||Trained|0.090|0.682|4.282|0.178|0.903|[Baidu](https://pan.baidu.com/s/1-Q8N1hPPjKz3BZXbPv_opw)|
|Res50|320x1024|Stereo|gpp|Trained|0.089|0.660|4.224|0.176|0.905|[Baidu](https://pan.baidu.com/s/1-Q8N1hPPjKz3BZXbPv_opw)|

**Results on NYU v2 test set** (trained on KITTI)
|Backbone|Resolution|Sup|PP|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|--------|----------|---|--|-----|--------|-------|----|-------|--|-----|
|Res50|288x384|Stereo||Reported|0.247|0.277|0.818|0.285|0.605|-|
|Res50|288x384|Stereo||Official|0.321|0.584|1.207|0.374|0.501|[Baidu](https://pan.baidu.com/s/1X4TWog23u2Wk6m6H_mbApA)|
|Res50|320x1024|Stereo||Trained|0.270|0.335|0.898|0.306|0.579|[Baidu](https://pan.baidu.com/s/1-Q8N1hPPjKz3BZXbPv_opw)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links is `smde`

### Evaluation Commands
For KITTI test sets.
```
# For Offical
python evaluate.py\
 --exp_opts options/EPCDepth/eval/EPCDepth-Res50_320_eval_OI.yaml\
 --model_path convert_models/EPCDepth_Res50_320_OI/model/epcdepth_res50_320.pth\
 -gpp # optional post-processing

# For Trained
python evaluate.py\
 --exp_opts options/EPCDepth/eval/EPCDepth-Res50_320_eval.yaml\
 --model_path convert_models/EPCDepth_Res50_320_bs6/model/best_model.pth\
 -gpp # optional post-processing
```

For NYU v2 test set
```
# For Offical
python evaluate.py\
 --exp_opts options/EPCDepth/eval/nyuv2/EPCDepth-Res50_320_eval.yaml\
 --model_path convert_models/EPCDepth_Res50_320_OI/model/epcdepth_res50_320.pth\
 --metric_name depth_nyu_mono # apply median scaling

# For Trained
python evaluate.py\
 --exp_opts options/EPCDepth/eval/nyuv2/EPCDepth-Res50_320_eval.yaml\
 --model_path convert_models/EPCDepth_Res50_320_bs6/model/best_model.pth\
 --metric_name depth_nyu_mono # apply median scaling
```