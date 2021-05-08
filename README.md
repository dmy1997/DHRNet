# DHRNet
DHRNet aims to generate different scale-sensitive weights (a parallel multi-branch architecture and add a soft conditional gate module) during feature fusion in HRNet thus improving the performance across different scales consistently on Citypersons.

![pipeline.png](https://github.com/dmy1997/DHRNet/blob/master/imgs/pipeline.PNG)


### Core files
The code base of our work is [MMDetection](https://github.com/open-mmlab/mmdetection). (Here is MMDetection-1.2.0)  
config: [configs/hrnet/faster_rcnn_hrnetv2p_w18_dynamic.py](https://github.com/dmy1997/DHRNet/blob/master/configs/hrnet/faster_rcnn_hrnetv2p_w18_dynamic.py)  
backbone: [DHRNet](https://github.com/dmy1997/DHRNet/blob/master/mmdet/models/backbones/dynamic_hrnet.py)


### Citation

```
@article{mmdetection,
  title   = {Learning a Dynamic High-Resolution Network for Multi-Scale Pedestrian Detection.},
  author  = {Mengyuan Ding, Shanshan Zhang, Jian Yang.},
  journal= {International Conference on Pattern Recognition (ICPR)},
  year={2020}
}
```

