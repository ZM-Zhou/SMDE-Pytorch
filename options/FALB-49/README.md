# FAL-Net
**Results on KITTI raw test set**
|Method|Info.|Sup|PP|Train|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|------|-----|---|--|-----|--------|-------|----|-------|--|-----|
|FAL-NetB(Stage1)|N=49+375x1242|Stereo||Official|0.105|0.681|4.521|0.194|0.870|[Baidu](https://pan.baidu.com/s/1g2aGl5Gp5G9cwrq_PCKalQ)|
|FAL-NetB(Stage1)|N=49+375x1242|Stereo||Trained|0.103|0.677|4.359|0.190|0.877|[Baidu](https://pan.baidu.com/s/17-4D_Lx-HHlRP2MWF5IlqQ)|
|FAL-NetB|N=49+375x1242|Stereo|mspp|Reported|0.093|0.564|3.973|0.174|0.898|-|
|FAL-NetB|N=49+375x1242|Stereo|mspp|Official|0.095|0.598|4.188|0.179|0.890|[Baidu](https://pan.baidu.com/s/1kN7hLqd0_c2yzufsOLypEA)|
|FAL-NetB|N=49+375x1242|Stereo||Trained|0.099|0.625|4.197|0.182|0.885|[Baidu](https://pan.baidu.com/s/1PhUJ_4s0nm41a49viZRczg)|
|FAL-NetB|N=49+375x1242|Stereo|mspp|Trained|0.096|0.604|4.185|0.179|0.889|[Baidu](https://pan.baidu.com/s/1PhUJ_4s0nm41a49viZRczg)|

**Results on KITTI improved test set**
|Method|Info.|Sup|PP|Train|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|------|-----|---|--|-----|--------|-------|----|-------|--|-----|
|FAL-NetB(Stage1)|N=49+375x1242|Stereo||Reported|0.076|0.331|3.167|0.116|0.932|-|
|FAL-NetB(Stage1)|N=49+375x1242|Stereo||Official|0.080|0.357|3.339|0.122|0.924|[Baidu](https://pan.baidu.com/s/1g2aGl5Gp5G9cwrq_PCKalQ)|
|FAL-NetB(Stage1)|N=49+375x1242|Stereo||Trained|0.079|0.352|3.136|0.118|0.932|[Baidu](https://pan.baidu.com/s/17-4D_Lx-HHlRP2MWF5IlqQ)|
|FAL-NetB|N=49+375x1242|Stereo|mspp|Reported|0.071|0.281|2.912|0.108|0.943|-|
|FAL-NetB|N=49+375x1242|Stereo|mspp|Official|0.072|0.296|2.994|0.109|0.941|[Baidu](https://pan.baidu.com/s/1kN7hLqd0_c2yzufsOLypEA)|
|FAL-NetB|N=49+375x1242|Stereo||Trained|0.076|0.317|3.002|0.114|0.936|[Baidu](https://pan.baidu.com/s/1PhUJ_4s0nm41a49viZRczg)|
|FAL-NetB|N=49+375x1242|Stereo|mspp|Trained|0.072|0.300|2.984|0.110|0.940|[Baidu](https://pan.baidu.com/s/1PhUJ_4s0nm41a49viZRczg)|

**Results on Make3D test set** (trained on KITTI)
|Backbone|Resolution|Sup|PP|Train|Abs Rel.|Sq Rel.|RMSE|log10|Model|
|--------|----------|---|--|-----|--------|-------|----|-------|-----|
|FAL-NetB|256x512|Stereo|mspp|Reported|0.284|2.803|6.643|-|
|FAL-NetB|256x512|Stereo|mspp|Official|0.283|2.724|6.697|0.148|[Baidu](https://pan.baidu.com/s/1kN7hLqd0_c2yzufsOLypEA)|
|FAL-NetB|256x512|Stereo||Trained|0.298|2.915|6.759|0.152|[Baidu](https://pan.baidu.com/s/1PhUJ_4s0nm41a49viZRczg)|
|FAL-NetB|256x512|Stereo|mspp|Trained|0.280|2.675|6.615|0.146|[Baidu](https://pan.baidu.com/s/1PhUJ_4s0nm41a49viZRczg)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* **code for all the download links is `smde`**
