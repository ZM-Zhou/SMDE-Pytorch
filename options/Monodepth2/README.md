# Monodepth2
**Results on KITTI raw test set**
|Backbone|Resolution|Sup|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|------|-----|---|-----|--------|-------|----|-------|--|--------|
|Res18|192x640|Mono|Reported|0.115|0.903|4.863|0.193|0.877|-|
|Res18|192x640|Mono|OI|0.115|0.909|4.866|0.193|0.877|[Baidu](https://pan.baidu.com/s/1bt9JHQnwClIuHk2RP2aD_g)|
|Res18|192x640|Stereo|Reported|0.109|0.873|4.960|0.209|0.864|-|
|Res18|192x640|Stereo|OI|0.109|0.875|4.959|0.209|0.864|[Baidu](https://pan.baidu.com/s/1EUwfWK89iOKcGa2SRo3-uw)|
|Res18|192x640|Stereo|Trained|0.107|0.861|4.910|0.204|0.867|[Baidu](https://pan.baidu.com/s/1ZGsjL8Mq-dhYwKNNqRU5MQ)|
|Res18|320x1024|Stereo|Reported|0.107|0.849|4.764|0.201|0.874|-|
|Res18|320x1024|Stereo|OI|0.107|0.851|4.765|0.201|0.874|[Baidu](https://pan.baidu.com/s/16cCslqM6Vdhye9QkuoCUSg)|
|Res18|320x1024|Stereo|Trained|0.106|0.798|4.700|0.202|0.871|[Baidu](https://pan.baidu.com/s/1Je1yhuYoa25eTUbS57kj4A)|

* `Reported` means that the results are reported in the paper.
* `OI` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links is `smde`