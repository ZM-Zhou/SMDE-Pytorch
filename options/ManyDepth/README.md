# ManyDepth
**Results on KITTI raw test set**
|Backbone|Resolution|Input|Sup|PP|Train|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|--------|----------|-----|---|--|-----|--------|-------|----|-------|--|-----|
|Res18|192x640|(-1,0)|Mono||Reported|0.098|0.770|4.459|0.176|0.900|-|
|Res18|192x640|(0)|Mono||Reported|0.118|0.892|4.764|0.192|0.871|-|
|Res18|192x640|(0)|Mono||Official|0.118|0.895|4.765|0.192|0.871|[Baidu](https://pan.baidu.com/s/168qFsk68t0117PXwcagtqQ)|

**Results on Cityscapes test set**
|Backbone|Resolution|Input|Sup|PP|Train|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|--------|----------|-----|---|--|-----|--------|-------|----|-------|--|-----|
|Res18|128x416|(-1,0)|Mono||Reported|0.114|1.193|6.223|0.170|0.875|-|
|Res18|128x416|(0)|Mono||Official|0.119|1.267|6.402|0.176|0.866|[Baidu](https://pan.baidu.com/s/1pX117dNZ_BzJh7lvwalA6w)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* code for all the download links is `smde`
