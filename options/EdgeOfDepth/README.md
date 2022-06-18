# Edge of Depth
**Results on KITTI raw test set**
|Backbone|Resolution|Sup|PP|Train|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|--------|----------|---|--|-----|--------|-------|----|-------|--|-----|
|Res50|320x1024|Stereo|gpp|Reported|0.091|0.646|4.244|0.177|0.898|-|
|Res50|320x1024|Stereo|gpp|Official|0.092|0.647|4.247|0.177|0.897|[Baidu](https://pan.baidu.com/s/1yToYiunNgNQZY8tunZOmGA)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links is `smde`

### Evaluation Commands
For KITTI test sets.
```
# For Offical
python evaluate.py\
 --exp_opts options/EdgeOfDepth/eval/EdgeOfDepth_320_eval_OI.yaml\
 --model_path convert_models/EdgeOfDepth_320_OI/model/EdgeOfDepth_320.pth\
 -gpp # optional post-processing
