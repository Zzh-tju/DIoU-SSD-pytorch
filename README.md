## Distance-IoU Loss into other SOTA detection methods can be found [here](https://github.com/Zzh-tju/DIoU). 

[[arxiv](https://arxiv.org/abs/1911.08287)] [[pdf](https://arxiv.org/pdf/1911.08287.pdf)]

### More improvement for this repo is coming soon. Faster and Better...

## SSD_FPN_DIoU,CIoU in PyTorch
The code references [SSD: Single Shot MultiBox Object Detector, in PyTorch](https://github.com/amdegroot/ssd.pytorch), [mmdet](https://github.com/open-mmlab/mmdetection) and [**JavierHuang**](https://github.com/JaryHuang). Currently, some experiments are carried out on the VOC dataset, if you want to train your own dataset, more details can be refer to the links above.

If you use this work, please consider citing:

```
@inproceedings{zheng2020distance,
  author    = {Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren},
  title     = {Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression},
  booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)},
   year      = {2020},
}
```

### Losses

Losses can be chosen with the `losstype` option in the `config/config.py` file The valid options are currently: `[Iou|Giou|Diou|Ciou|SmoothL1]`.

```
VOC:
  'losstype': 'Ciou'
```
### DIoU-NMS
NMS can be chosen with the `nms_kind` option in the `config/config.py` file. If set it to `greedynms`, it means using greedy-NMS.
Besides that, similar to DIoU-NMS in Faster R-CNN, we also introduce `beta1` for DIoU-NMS in SSD, that is DIoU = IoU - R_DIoU ^ {beta1}. With this operation, DIoU-NMS may perform better than default beta1=1.0. I recommand for SSD `beta1=0.5~1.0` will be good.
```
  'nms_kind': "diounms"
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
|IoU|51.01|54.74|
|GIoU|51.06|55.48|
|DIoU|51.31|55.71|
|CIoU|51.44|56.16|

## Pretrained weights

Here are the trained models using the configurations in this repository.

 - [IoU](https://pan.baidu.com/s/1eNcD9CrnRL79VIH5lsOTPA)
 - [GIoU](https://pan.baidu.com/s/1_b1RS5qaRVJUwi27mcpXow)
 - [DIoU](https://pan.baidu.com/s/1x1keVP958-DyN_OuWdDAXA)
 - [CIoU](https://pan.baidu.com/s/10sodf37QjTVMEzOIVD8cNA)
