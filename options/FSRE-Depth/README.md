# FSRE-Depth
**Results on KITTI raw test set**
|Backbone|Resolution|Sup|PP|Train|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|--------|----------|---|--|-----|--------|-------|----|-------|--|-----|
|Res18|192x640|Mono||Reported|0.105|0.722|4.547|0.182|0.886|-|
|Res18|192x640|Mono||Official|0.105|0.711|4.546|0.182|0.886|[Baidu](https://pan.baidu.com/s/1u9VhbIPN67E12oqLSFzGiA)/[Googke](https://drive.google.com/file/d/1ouVCR_vWE8XZUccE6lkLAgZ92vqvFY7M/view?usp=sharing)|
|Res18|192x640|Mono||Trained|0.107|0.751|4.525|0.182|0.886|[Baidu](https://pan.baidu.com/s/1jFDl9ofeFqxBQGyuGsn1WQ)/[Google](https://drive.google.com/file/d/18ssmBZ9F7p3X7pYAkPWfzWBeUlDXFoWr/view?usp=sharing)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links of pan Baidu is `smde`.

## Training Note
For training FSRE-Depth in this repo, please first download the pre-computed semantic segmentation label provided in thier [official repo](https://github.com/hyBlue/FSRE-Depth) and unzip it in the root of kitti dataset.
