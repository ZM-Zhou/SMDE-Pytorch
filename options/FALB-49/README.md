# FAL-Net
**Results on KITTI raw test set**
|Method|Info.|Sup|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|------|-----|---|-----|--------|-------|----|-------|--|--------|
|FAL-NetB(Stage1)|N=49+375x1242|Stereo|OI|0.105|0.681|4.521|0.194|0.870|[Baidu](https://pan.baidu.com/s/1g2aGl5Gp5G9cwrq_PCKalQ)|
|FAL-NetB|N=49+375x1242|Stereo|Reported|0.097|0.590|3.991|0.177|0.893|-|
|FAL-NetB|N=49+375x1242|Stereo|OI|0.099|0.624|4.204|0.184|0.884|[Baidu](https://pan.baidu.com/s/1kN7hLqd0_c2yzufsOLypEA)|

**Results on KITTI improved test set**
|Method|Info.|Sup|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|------|-----|---|-----|--------|-------|----|-------|--|--------|
|FAL-NetB(Stage1)|N=49+375x1242|Stereo|Reported|0.076|0.331|3.167|0.116|0.932|-|
|FAL-NetB(Stage1)|N=49+375x1242|Stereo|OI|0.080|0.357|3.339|0.122|0.924|[Baidu](https://pan.baidu.com/s/1g2aGl5Gp5G9cwrq_PCKalQ)|
|FAL-NetB|N=49+375x1242|Stereo|Reported|0.075|0.298|2.905|0.112|0.937|-|
|FAL-NetB|N=49+375x1242|Stereo|OI|0.075|0.312|3.005|0.114|0.935|[Baidu](https://pan.baidu.com/s/1kN7hLqd0_c2yzufsOLypEA)|

* `Reported` means that the results are reported in the paper.
* `OI` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* **code for all the download links is `smde`**
