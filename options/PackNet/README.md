# PackNet
**Results on KITTI raw test set**
|Backbone|Resolution|Sup|PP|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|--------|----------|---|--|-----|--------|-------|----|-------|--|-----|
|PackNet|192x640|Mono||Reported|0.111|0.785|4.601|0.189|0.878|-|
|PackNet|192x640|Mono||Official|0.110|0.831|4.648|0.187|0.881|[Baidu](https://pan.baidu.com/s/1d_uL1q2_bsGEskFDcEfBGA)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links is `smde`

### Evaluation Commands
For KITTI test sets.
```
# For Offical
python evaluate.py\
 --exp_opts options/PackNet/eval/Packnet_M_192_eval.yaml\
 --model_path convert_models/PackNet_M_192_OI/model/PackNet_M_192.pth\
 --metric_name depth_kitti_mono
