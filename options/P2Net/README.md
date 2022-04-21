# P2Net
**Results on NYU v2 test set**
|Backbone|Resolution|Sup|PP|Train|Abs Rel.|Abs Sq.|RMSE|log10|A1|Model|
|--------|----------|---|--|-----|--------|-------|----|-------|-----|--|-----|
|Res18(3f)|288x384|Mono||Reported|0.159|-|0.599|0.068|0.772|-|
|Res18(5f)|288x384|Mono||Reported|0.150|-|0.561|0.064|0.796|-|
|Res18(5f)|288x384|Mono|gpp|Reported|0.147|-|0.553|0.062|0.801|-|
|Res18(5f)|288x384|Mono||Official|0.149|-|0.556|0.063|0.797|[Baidu](https://pan.baidu.com/s/1wpN6O-e453e9n8AJqvG3JA)|
|Res18(5f)|288x384|Mono|gpp|Official|0.146|-|0.545|0.062|0.804|[Baidu](https://pan.baidu.com/s/1wpN6O-e453e9n8AJqvG3JA)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links is `smde`

### Evaluation Commands
For NYUv2 test set.
```
# For Offical
python evaluate.py\
 --exp_opts options/P2Net/eval/nyuv2/P2Net_288_eval.yaml\
 --model_path convert_models/P2Net_NYUv2_288_OI/model/P2Net_NYUv2_288.pth\
 --metric_name depth_nyu_mono\
 -gpp # optional post-processing
