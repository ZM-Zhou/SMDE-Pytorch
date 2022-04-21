# PackNet
**Results on KITTI raw test set**
|Backbone|Resolution|Sup|PP|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|--------|----------|---|--|-----|--------|-------|----|-------|--|-----|
|PackNet|192x640|Mono||Reported|0.111|0.785|4.601|0.189|0.878|-|
|PackNet|192x640|Mono||Official|0.110|0.831|4.648|0.187|0.881|[Baidu](https://pan.baidu.com/s/1d_uL1q2_bsGEskFDcEfBGA)|
|PackNet|384x1280|Mono(v/CS+K)||Reported|0.103|0.796|4.404|0.189|0.881|-|
|PackNet|384x1280|Mono(v/CS+K)||Official|0.102|0.855|4.701|0.192|0.882|[Baidu](https://pan.baidu.com/s/1rgXq2ybBZqN3vPnAt8yuZA)|

**Results on KITTI improved test set**
|Backbone|Resolution|Sup|PP|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|--------|----------|---|--|-----|--------|-------|----|-------|--|-----|
|PackNet|192x640|Mono||Reported|0.078|0.420|3.485|0.121|0.931|-|
|PackNet|192x640|Mono||Official|0.079|0.442|3.578|0.123|0.928|[Baidu](https://pan.baidu.com/s/1d_uL1q2_bsGEskFDcEfBGA)|
|PackNet|384x1280|Mono(v/CS+K)||Reported|0.075|0.420|3.293|0.114|0.938|-|
|PackNet|384x1280|Mono(v/CS+K)||Official|0.085|0.458|3.526|0.126|0.928|[Baidu](https://pan.baidu.com/s/1rgXq2ybBZqN3vPnAt8yuZA)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links is `smde`

### Evaluation Commands
For KITTI test sets.
```
# For Offical in 192x640
python evaluate.py\
 --exp_opts options/PackNet/eval/PackNet_M_192_eval.yaml\
 --model_path convert_models/PackNet_M_192_OI/model/PackNet_M_192.pth\
 --metric_name depth_kitti_mono

# For Offical in 384x1280 (scale aware)
python evaluate.py\
 --exp_opts options/PackNet/eval/PackNet_M_384_eval.yaml\
 --model_path convert_models/PackNet_Mv_CS+K_384_OI/model/PackNet_Mv_CS+K_384.pth

# For Offical in 192x640 (Improved)
python evaluate.py\
 --exp_opts options/PackNet/eval/PackNet_M_192_eval_improved.yaml\
 --model_path convert_models/PackNet_M_192_OI/model/PackNet_M_192.pth\
 --metric_name depth_kitti_mono

# For Offical in 384x1280 (scale aware) (Improved)
python evaluate.py\
 --exp_opts options/PackNet/eval/PackNet_M_384_eval_improved.yaml\
 --model_path convert_models/PackNet_Mv_CS+K_384_OI/model/PackNet_Mv_CS+K_384.pth
```
