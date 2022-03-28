# EPCDepth
**Results on KITTI raw test set**
|Backbone|Resolution|PP|Sup|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|--------|----------|--|---|-----|--------|-------|----|-------|--|-----|
|Res50|320x1024|pp|Stereo|Reported|0.091|0.646|4.207|0.176|0.901|-|
|Res50|320x1024||Stereo|OI|0.096|0.684|4.278|0.184|0.889|[Baidu](https://pan.baidu.com/s/1X4TWog23u2Wk6m6H_mbApA)|
|Res50|320x1024|pp|Stereo|OI|0.095|0.667|4.239|0.183|0.889|[Baidu](https://pan.baidu.com/s/1X4TWog23u2Wk6m6H_mbApA)|

* `Reported` means that the results are reported in the paper.
* `OI` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links is `smde`