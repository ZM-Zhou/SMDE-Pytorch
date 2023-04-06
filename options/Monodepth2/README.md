# Monodepth2
**Results on KITTI raw test set (Mono)**
|Backbone|Resolution|Sup|Train|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|------|-----|---|-----|--------|-------|----|-------|--|--------|
|Res18|192x640|Mono|Reported|0.115|0.903|4.863|0.193|0.877|-|
|Res18|192x640|Mono|Official|0.115|0.909|4.866|0.193|0.877|[Baidu](https://pan.baidu.com/s/1bt9JHQnwClIuHk2RP2aD_g)/[Google](https://drive.google.com/file/d/1Vcw3LRsHH6G3xXQmF4kFMAGTzgqsm-8T/view?usp=share_link)|
|Res18|192x640|Mono|Trained|0.113|0.858|4.753|0.190|0.879|[Baidu](https://pan.baidu.com/s/1eTZa2-5Kd9TJNJOJvDzetg)/[Google](https://drive.google.com/file/d/1MglGKbiFGRW7PwJmX8Y3xYhUDbXa5UBf/view?usp=share_link)|
|Res18|320x1024|Mono|Reported|0.115|0.882|4.701|0.190|0.879|-|
|Res18|320x1024|Mono|Official|0.115|0.886|4.704|0.190|0.879|[Baidu](https://pan.baidu.com/s/1d94jQ-XNaJNviVDBu7p7BA)/[Google](https://drive.google.com/file/d/1UM3ZG7e4RSaEb7m8ycMOYeK_IUrGcIhh/view?usp=share_link)|
|Res18|320x1024|Mono|Trained|0.109|0.797|4.533|0.184|0.887|[Baidu](https://pan.baidu.com/s/1T3IGfBB2c5Y2xskACRg3aQ)/[Google](https://drive.google.com/file/d/1FZgXmeCRz5RHTEZarQKRAJBxklblQT5a/view?usp=sharing)|

**Results on KITTI raw test set (Stereo)**
|Backbone|Resolution|Sup|Train|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|Model|
|------|-----|---|-----|--------|-------|----|-------|--|--------|
|Res18|192x640|Stereo|Reported|0.109|0.873|4.960|0.209|0.864|-|
|Res18|192x640|Stereo|Official|0.109|0.875|4.959|0.209|0.864|[Baidu](https://pan.baidu.com/s/1EUwfWK89iOKcGa2SRo3-uw)/[Google](https://drive.google.com/file/d/14_iEPjqUN5_ZuUc5cEd6DU_Lh2T87Mc2/view?usp=share_link)|
|Res18|192x640|Stereo|Trained|0.108|0.858|4.819|0.202|0.866|[Baidu](https://pan.baidu.com/s/1gwWUzUKNTWq5MuUzUzhJdg)/[Google](https://drive.google.com/file/d/1SkmyaO6U3xvNf7dUAfHXqA00x2zjzNyq/view?usp=sharing)|
|Res18|320x1024|Stereo|Reported|0.107|0.849|4.764|0.201|0.874|-|
|Res18|320x1024|Stereo|Official|0.107|0.851|4.765|0.201|0.874|[Baidu](https://pan.baidu.com/s/16cCslqM6Vdhye9QkuoCUSg)/[Google](https://drive.google.com/file/d/1LggauX1CnoXT0PpHz-3lq80IYWuFQFY-/view?usp=sharing)|
|Res18|320x1024|Stereo|Trained|0.104|0.824|4.747|0.200|0.875|[Baidu](https://pan.baidu.com/s/1Kj9HOo15murscIsOchMEUA)/[Google](https://drive.google.com/file/d/1tpkukOfOzMF2lv56ZkdxPuYwjFwBtyfC/view?usp=share_link)|

**Results on Make3D test set** (trained on KITTI)
|Backbone|Resolution|Sup|PP|Train|Abs Rel.|Sq Rel.|RMSE|log10|Model|
|--------|----------|---|--|-----|--------|-------|----|-------|-----|
|Res18|192x640|Stereo||Reported|0.322|3.589|7.417|0.163|
|Res18|192x640|Stereo||Official|0.321|3.379|7.254|0.163|[Baidu](https://pan.baidu.com/s/1EUwfWK89iOKcGa2SRo3-uw)/[Google](https://drive.google.com/file/d/14_iEPjqUN5_ZuUc5cEd6DU_Lh2T87Mc2/view?usp=share_link)|
|Res18|192x640|Stereo||Trained|0.311|3.166|6.959|0.158|[Baidu](https://pan.baidu.com/s/1gwWUzUKNTWq5MuUzUzhJdg)/[Google](https://drive.google.com/file/d/1SkmyaO6U3xvNf7dUAfHXqA00x2zjzNyq/view?usp=sharing)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links of pan Baidu is `smde`.
