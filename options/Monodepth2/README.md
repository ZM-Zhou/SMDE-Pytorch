# Monodepth2
**Results on KITTI raw test set**
|Backbone|Resolution|Sup|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|------|-----|---|-----|--------|-------|----|-------|--|--------|
|Res18|192x640|Mono|Reported|0.115|0.903|4.863|0.193|0.877|-|
|Res18|192x640|Mono|Official|0.115|0.909|4.866|0.193|0.877|[Baidu](https://pan.baidu.com/s/1bt9JHQnwClIuHk2RP2aD_g)|
|Res18|192x640|Mono|Trained|0.113|0.858|4.753|0.190|0.879|[Baidu](https://pan.baidu.com/s/1eTZa2-5Kd9TJNJOJvDzetg)|
|Res18|320x1024|Mono|Reported|0.115|0.882|4.701|0.190|0.879|-|
|Res18|320x1024|Mono|Official|0.115|0.886|4.704|0.190|0.879|[Baidu](https://pan.baidu.com/s/1d94jQ-XNaJNviVDBu7p7BA)|
|Res18|320x1024|Mono|Trained|0.109|0.797|4.533|0.184|0.888|[Baidu](https://pan.baidu.com/s/1T3IGfBB2c5Y2xskACRg3aQ)|
|Res18|192x640|Stereo|Reported|0.109|0.873|4.960|0.209|0.864|-|
|Res18|192x640|Stereo|Official|0.109|0.875|4.959|0.209|0.864|[Baidu](https://pan.baidu.com/s/1EUwfWK89iOKcGa2SRo3-uw)|
|Res18|192x640|Stereo|Trained|0.108|0.858|4.819|0.202|0.866|[Baidu](https://pan.baidu.com/s/1gwWUzUKNTWq5MuUzUzhJdg)|
|Res18|320x1024|Stereo|Reported|0.107|0.849|4.764|0.201|0.874|-|
|Res18|320x1024|Stereo|Official|0.107|0.851|4.765|0.201|0.874|[Baidu](https://pan.baidu.com/s/16cCslqM6Vdhye9QkuoCUSg)|
|Res18|320x1024|Stereo|Trained|0.104|0.824|4.747|0.200|0.875|[Baidu](https://pan.baidu.com/s/1Kj9HOo15murscIsOchMEUA)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links is `smde`