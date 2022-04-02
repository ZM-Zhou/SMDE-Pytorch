# DepthHints
**Results on KITTI raw test set**
|Backbone|Resolution|Sup|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|------|-----|---|-----|--------|-------|----|-------|--|--------|
|Res50|320x1024|Stereo|Reported|0.096|0.710|4.393|0.185|0.890|-|
|Res50|320x1024|Stereo|Official|0.097|0.737|4.448|0.186|0.889|[Baidu](https://pan.baidu.com/s/1OPesveOI0us8rVEwal-pGg)|
|Res50|320x1024|Stereo|Trained|0.094|0.680|4.333|0.181|0.894|[Baidu](https://pan.baidu.com/s/12xv0IY_hcO1YtsEZJ2Vuog)|

**Results on KITTI improved test set**
|Backbone|Resolution|Sup|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|------|-----|---|-----|--------|-------|----|-------|--|--------|
|Res50|320x1024|Stereo|Official|0.070|0.336|3.095|0.109|0.943|[Baidu](https://pan.baidu.com/s/1OPesveOI0us8rVEwal-pGg)|
|Res50|320x1024|Stereo|Trained|0.070|0.323|3.028|0.107|0.944|[Baidu](https://pan.baidu.com/s/12xv0IY_hcO1YtsEZJ2Vuog)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links is `smde`