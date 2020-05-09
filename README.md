<img src="CIoU.png" width="800px"/>

## Complete-IoU Loss and Cluster-NMS for improving Object Detection and Instance Segmentation. 

This is the code for our papers:
 - [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)
 - [Enhancing Geometric Factors into Model Learning and Inference for Object Detection and Instance Segmentation](https://arxiv.org/abs/2005.03572)

```
@Inproceedings{zheng2020distance,
  author    = {Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren},
  title     = {Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression},
  booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)},
   year      = {2020},
}

@Article{zheng2020ciou,
  author= {Zhaohui Zheng, Ping Wang, Dongwei Ren, Wei Liu, Rongguang Ye, Qinghua Hu, Wangmeng Zuo},
  title={Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation},
  journal={arXiv:2005.03572},
  year={2020}
}
```

## SSD_FPN_DIoU,CIoU in PyTorch
The code references [SSD: Single Shot MultiBox Object Detector, in PyTorch](https://github.com/amdegroot/ssd.pytorch), [mmdet](https://github.com/open-mmlab/mmdetection) and [**JavierHuang**](https://github.com/JaryHuang). Currently, some experiments are carried out on the VOC dataset, if you want to train your own dataset, more details can be refer to the links above.

### Losses

Losses can be chosen with the `losstype` option in the `config/config.py` file The valid options are currently: `[Iou|Giou|Diou|Ciou|SmoothL1]`.

```
VOC:
  'losstype': 'Ciou'
```

## Fold-Structure
The fold structure as follow:
- config/
	- config.py
	- __init__.py
- data/
	- __init__.py
 	- VOC.py
	- VOCdevkit/
- model/
	- build_ssd.py
	- __init__.py
	- backbone/
	- neck/
	- head/
	- utils/
- utils/
	- box/
	- detection/
	- loss/
	- __init__.py
- tools/
	- train.py
	- eval.py
	- test.py
- work_dir/
	

## Environment
- pytorch 0.4.1
- python3+
- visdom 
	- for real-time loss visualization during training!
	```Shell
	pip install visdom
	```
	- Start the server (probably in a screen or tmux)
	```Shell
	python visdom
	```
  * Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).


## Datasets
- PASCAL VOC:Download VOC2007, VOC2012 dataset, then put VOCdevkit in the data directory


## Training

### Training VOC
- The pretrained model refer [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch),you can download it.

- In the DIoU-SSD-pytorch fold:
```Shell
python tools/train.py
```

- Note:
  * For training, default NVIDIA GPU.
  * You can set the parameters in the train.py (see 'tools/train.py` for options) 
  * In the config,you can set the work_dir to save your training weight.(see 'configs/config.py`) 

## Evaluation
- To evaluate a trained network:

```Shell
python tools/ap.py --trained_model {your_weight_address}
```

For example: (the output is AP50, AP75 and AP of our CIoU loss)
```
Results:
0.033
0.015
0.009
0.011
0.008
0.083
0.044
0.042
0.004
0.014
0.026
0.034
0.010
0.006
0.009
0.006
0.009
0.013
0.106
0.011
0.025
~~~~~~~~

--------------------------------------------------------------
Results computed with the **unofficial** Python eval code.
Results should be very close to the official MATLAB eval code.
--------------------------------------------------------------
0.7884902583981603 0.5615516772893671 0.5143832356646468
```

## Test
- To test a trained network:

```Shell
python test.py -- trained_model {your_weight_address}
```
if you want to visual the box, you can add the command --visbox True(default False)

## Performance

#### VOC2007 Test mAP
- Backbone is ResNet50-FPN:

| Test |AP|AP75|
|:-:|:-:|:-:|
|IoU|51.0|54.7|
|GIoU|51.1|55.4|
|DIoU|51.3|55.7|
|CIoU|51.5|56.4|
|CIoU 16|53.3|58.2|

##### "16" means bbox regression weight is set to 16.
## Cluster-NMS

See `Detect` function of [utils/detection/detection.py](utils/detection/detection.py) for our Cluster-NMS implementation.

Currently, NMS only surports `cluster_nms`, `cluster_diounms`, `cluster_weighted_nms`, `cluster_weighted_diounms`. (See `'nms_kind'` in [config/config.py](config/config.py))

#### Hardware
 - 1 RTX 2080 Ti
 - Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz

| Backbone  | Loss  | Regression weight  | NMS  | FPS  | time | box AP | box AP75 |
|:-------------:|:-------:|:-------:|:------------------------------------:|:----:|:----:|:----:|:----:|
| Resnet50-FPN | CIoU  | 5  |          Fast NMS           |**28.8**|**34.7**|  50.7  |  56.2  |
| Resnet50-FPN | CIoU  | 5  |        Original NMS         |  17.8  |  56.1  |  51.5  |  56.4  |
| Resnet50-FPN | CIoU  | 5  |         DIoU-NMS            |  11.4  |  87.6  |  51.9  |  56.6  |
| Resnet50-FPN | CIoU  | 5  |        Cluster-NMS          |  28.0  |  35.7  |  51.5  |  56.4  |
| Resnet50-FPN | CIoU  | 5  |      Cluster-DIoU-NMS       |  27.7  |  36.1  |  51.9  |  56.6  |
| Resnet50-FPN | CIoU  | 5  |     Weighted Cluster-NMS    |  26.8  |  37.3  |  51.9  |  56.3  |
| Resnet50-FPN | CIoU  | 5  | Weighted + Cluster-DIoU-NMS |  26.5  |  37.8  |**52.4**|**57.0**|

#### Hardware
 - 1 RTX 2080
 - Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz

| Backbone  | Loss  | Regression weight  | NMS  | FPS  | time | box AP | box AP75 |
|:-------------:|:-------:|:-------:|:------------------------------------:|:----:|:----:|:----:|:----:|
| Resnet50-FPN | CIoU  | 16  |        Original NMS         |  19.7  |  50.9  |  53.3  |  58.2  |
| Resnet50-FPN | CIoU  | 16  |        Cluster-NMS          |  28.0  |  35.7  |  53.4  |  58.2  |
| Resnet50-FPN | CIoU  | 16  |      Cluster-DIoU-NMS       |  26.5  |  37.7  |  53.7  |  58.6  |
| Resnet50-FPN | CIoU  | 16  |     Weighted Cluster-NMS    |  26.9  |  37.2  |  53.8  |  58.7  |
| Resnet50-FPN | CIoU  | 16  | Weighted + Cluster-DIoU-NMS |  26.3  |  38.0  |**54.1**|**59.0**|
#### Note:
 - Here the box coordinate weighted average is only performed in `IoU> 0.8`. We searched that `IoU > NMS thresh` is not good for SSD and `IoU>0.9` is almost same to `Cluster-NMS`. (Refer to [CAD](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8265304) for the details of Weighted-NMS.)
 
 - We further incorporate DIoU into Weighted Cluster-NMS for SSD which can get higher AP.
  
 - Note that Torchvision NMS has the fastest speed, that is owing to CUDA implementation and engineering accelerations (like upper triangular IoU matrix only). However, our Cluster-NMS requires less iterations for NMS and can also be further accelerated by adopting engineering tricks.
 
 - Currently, Torchvision NMS use IoU as criterion, not DIoU. However, if we directly replace IoU with DIoU in Original NMS, it will costs much more time due to the sequence operation. Now, Cluster-DIoU-NMS will significantly speed up DIoU-NMS and obtain exactly the same result.
 
 - Torchvision NMS is a function in Torchvision>=0.3, and our Cluster-NMS can be applied to any projects that use low version of Torchvision and other deep learning frameworks as long as it can do matrix operations. **No other import, no need to compile, less iteration, fully GPU-accelerated and better performance**.
## Pretrained weights

Here are the trained models using the configurations in this repository.

 - [IoU bbox regression weight 5](https://pan.baidu.com/s/1eNcD9CrnRL79VIH5lsOTPA)
 - [GIoU bbox regression weight 5](https://pan.baidu.com/s/1_b1RS5qaRVJUwi27mcpXow)
 - [DIoU bbox regression weight 5](https://pan.baidu.com/s/1x1keVP958-DyN_OuWdDAXA)
 - [CIoU bbox regression weight 5](https://share.weiyun.com/5LSzur7)
 - [CIoU bbox regression weight 16](https://share.weiyun.com/5U3OHez)
