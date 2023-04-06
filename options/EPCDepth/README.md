# EPCDepth
**Results on KITTI raw test set**
|Backbone|Resolution|Sup|PP|Train|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|--------|----------|---|--|-----|--------|-------|----|-------|--|-----|
|Res50|320x1024|Stereo|gpp|Reported|0.091|0.646|4.207|0.176|0.901|-|
|Res50|320x1024|Stereo|gpp|Official|0.095|0.667|4.239|0.183|0.889|[Baidu](https://pan.baidu.com/s/1X4TWog23u2Wk6m6H_mbApA)/[Google](https://drive.google.com/file/d/1AnIEyKgcTGxtQs61a-NSg-ai9jTPfixL/view?usp=sharing)|
|Res50|320x1024|Stereo||Trained|0.090|0.682|4.282|0.178|0.903|[Baidu](https://pan.baidu.com/s/1-Q8N1hPPjKz3BZXbPv_opw)/[Google](https://drive.google.com/file/d/11VmegFBMOhB5eGpL49izIYqgf4zeNd4i/view?usp=sharing)|
|Res50|320x1024|Stereo|gpp|Trained|0.089|0.660|4.224|0.176|0.905|[Baidu](https://pan.baidu.com/s/1-Q8N1hPPjKz3BZXbPv_opw)/[Google](https://drive.google.com/file/d/11VmegFBMOhB5eGpL49izIYqgf4zeNd4i/view?usp=sharing)|

**Results on NYU v2 test set** (trained on KITTI)
|Backbone|Resolution|Sup|PP|Train|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|--------|----------|---|--|-----|--------|-------|----|-------|--|-----|
|Res50|288x384|Stereo|gpp|Reported|0.247|0.277|0.818|0.285|0.605|-|
|Res50|288x384|Stereo|gpp|Official|0.280|0.371|0.972|0.335|0.530|[Baidu](https://pan.baidu.com/s/1X4TWog23u2Wk6m6H_mbApA)/[Google](https://drive.google.com/file/d/1AnIEyKgcTGxtQs61a-NSg-ai9jTPfixL/view?usp=sharing)|
|Res50|288x384|Stereo||Trained|0.261|0.301|0.851|0.299|0.591|[Baidu](https://pan.baidu.com/s/1-Q8N1hPPjKz3BZXbPv_opw)/[Google](https://drive.google.com/file/d/11VmegFBMOhB5eGpL49izIYqgf4zeNd4i/view?usp=sharing)|
|Res50|288x384|Stereo|gpp|Trained|0.258|0.293|0.841|0.296|0.596|[Baidu](https://pan.baidu.com/s/1-Q8N1hPPjKz3BZXbPv_opw)/[Google](https://drive.google.com/file/d/11VmegFBMOhB5eGpL49izIYqgf4zeNd4i/view?usp=sharing)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links of pan Baidu is `smde`.
