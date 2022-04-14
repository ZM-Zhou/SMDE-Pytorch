# DepthHints
## Evaluation Results
**Results on KITTI raw test set**
|Backbone|Resolution|Sup|PP.|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|--------|----------|---|---|-----|--------|-------|----|-------|--|-----|
|Res50|320x1024|Stereo|gpp|Reported|0.096|0.710|4.393|0.185|0.890|-|
|Res50|320x1024|Stereo|gpp|Official|0.096|0.714|4.395|0.185|0.890|[Baidu](https://pan.baidu.com/s/1OPesveOI0us8rVEwal-pGg)|
|Res50|320x1024|Stereo||Trained|0.094|0.680|4.333|0.181|0.894|[Baidu](https://pan.baidu.com/s/12xv0IY_hcO1YtsEZJ2Vuog)|
|Res50|320x1024|Stereo|gpp|Trained|0.093|0.665|4.298|0.179|0.895|[Baidu](https://pan.baidu.com/s/12xv0IY_hcO1YtsEZJ2Vuog)|

**Results on KITTI improved test set**
|Backbone|Resolution|Sup|PP.|Train|Abs Rel.|Abs Sq.|RMSE|RMSElog|A1|Model|
|--------|----------|---|---|-----|--------|-------|----|-------|--|-----|
|Res50|320x1024|Stereo|gpp|Official|0.069|0.324|3.043|0.108|0.944|[Baidu](https://pan.baidu.com/s/1OPesveOI0us8rVEwal-pGg)|
|Res50|320x1024|Stereo||Trained|0.070|0.323|3.028|0.107|0.944|[Baidu](https://pan.baidu.com/s/12xv0IY_hcO1YtsEZJ2Vuog)|
|Res50|320x1024|Stereo|gpp|Trained|0.070|0.313|2.990|0.106|0.946|[Baidu](https://pan.baidu.com/s/12xv0IY_hcO1YtsEZJ2Vuog)|

* `Reported` means that the results are reported in the paper.
* `Official` means that the results are predicted with the models got from their Official Implementations.
* `Trained` means that the results are predicted with the models trained with this repository.
* code for all the download links is `smde`

### Evaluation Commands
For KITTI test sets.
```
# For Offical
python evaluate.py\
 --exp_opts options/DepthHints/eval/DepthHints_S_320_eval_OI.yaml\
 --model_path convert_models/DepthHints_Res50_320_OI/model/depthhints_res50_320.pth\
 -gpp # optional post-processing

# For Trained
python evaluate.py\
 --exp_opts options/DepthHints/eval/DepthHints_S_320_eval.yaml\
 --model_path convert_models/DepthHints_Res50_320_bs6/model/best_model.pth\
 -gpp # optional post-processing

# For Offical (Improved)
python evaluate.py\
 --exp_opts options/DepthHints/eval/DepthHInts_S_320_eval_improved_OI.yaml\
 --model_path convert_models/DepthHints_Res50_320_OI/model/depthhints_res50_320.pth\
 -gpp # optional post-processing

# For Trained (Improved)
python evaluate.py\
 --exp_opts  options/DepthHints/eval/DepthHints_S_320_eval_improved.yaml\
 --model_path convert_models/DepthHints_Res50_320_bs6/model/best_model.pth\
 -gpp # optional post-processing
```
