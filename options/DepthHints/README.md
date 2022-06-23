# DepthHints
## Evaluation Results
**Results on KITTI raw test set**
|Backbone|Resolution|Sup|PP.|Train|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|--------|----------|---|---|-----|--------|-------|----|-------|--|-----|
|Res50|192x640|Stereo|gpp|Reported|0.102|0.762|4.602|0.189|0.880|-|
|Res50|192x640|Stereo|gpp|Official|0.102|0.764|4.603|0.189|0.880|[Baidu](https://pan.baidu.com/s/1NAeyKkk15C0OLfNRigQGVQ)|
|Res50|192x640|Stereo||Trained|0.100|0.757|4.586|0.188|0.883|[Baidu](https://pan.baidu.com/s/11BEdrxRFIzgIU3DHjW1ilw)|
|Res50|192x640|Stereo|gpp|Trained|0.100|0.748|4.550|0.187|0.884|[Baidu](https://pan.baidu.com/s/11BEdrxRFIzgIU3DHjW1ilw)|
|Res50|320x1024|Stereo|gpp|Reported|0.096|0.710|4.393|0.185|0.890|-|
|Res50|320x1024|Stereo|gpp|Official|0.096|0.714|4.395|0.185|0.890|[Baidu](https://pan.baidu.com/s/1OPesveOI0us8rVEwal-pGg)|
|Res50|320x1024|Stereo||Trained|0.094|0.680|4.333|0.181|0.894|[Baidu](https://pan.baidu.com/s/12xv0IY_hcO1YtsEZJ2Vuog)|
|Res50|320x1024|Stereo|gpp|Trained|0.093|0.665|4.298|0.179|0.895|[Baidu](https://pan.baidu.com/s/12xv0IY_hcO1YtsEZJ2Vuog)|

**Results on KITTI improved test set**
|Backbone|Resolution|Sup|PP.|Train|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|--------|----------|---|---|-----|--------|-------|----|-------|--|-----|
|Res50|320x1024|Stereo|gpp|Official|0.069|0.324|3.047|0.108|0.944|[Baidu](https://pan.baidu.com/s/1OPesveOI0us8rVEwal-pGg)|
|Res50|320x1024|Stereo||Trained|0.070|0.323|3.028|0.107|0.944|[Baidu](https://pan.baidu.com/s/12xv0IY_hcO1YtsEZJ2Vuog)|
|Res50|320x1024|Stereo|gpp|Trained|0.070|0.313|2.990|0.106|0.946|[Baidu](https://pan.baidu.com/s/12xv0IY_hcO1YtsEZJ2Vuog)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links is `smde`
