# ZeroWaste3D Approach  on Instance Segmentation 

This section details the approach used for instnace segmentaiton methods utilizing `Mask-RCNN` + `U-Net` | `Yolact` architectures for a fully connected convolutional model developed to perform real time object segmentation using RGB images. 

We utilize [Detectron2](https://github.com/facebookresearch/detectron2) as the backend for training and testing DNN models as well as feature extracture with with goal of producint an accurate detector for instance segmentation.


## Table of Contents
- [ZeroWaste3D Approach  on Instance Segmentation](#zerowaste3d-approach-on-instance-segmentation)
  - [Table of Contents](#table-of-contents)
    - [Preqequisites](#preqequisites)
      - [Requirements](#requirements)
      - [Installation](#installation)
    - [ZeroWaste COCO Dataset - Setup](#zerowaste-coco-dataset---setup)
  - [Getting Started](#getting-started)
    - [Get pretrained model](#get-pretrained-model)
  - [Training](#training)
    - [Training - Quickstart](#training---quickstart)
    - [Training from scratch](#training-from-scratch)
    - [Training from pretrained checkpoint](#training-from-pretrained-checkpoint)
    - [Troubleshooting](#troubleshooting)
      - [Mask RCNN Errors](#mask-rcnn-errors)
    - [References](#references)

### Preqequisites

#### Requirements
* [Python](https://www.python.org/downloads/) > 3.5
* [Pytorch](http://pytorch.org/) > 1.3
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
* [Cuda](https://developer.nvidia.com/cuda-toolkit) > 10.0
* [Detectron2](https://github.com/facebookresearch/detectron2)
* OpenCV, needed by demo and visualization


#### Installation

TODO

```
# from ZeroWaste3D/
conda activate cv
git submodule update --init --recursive
```

Installing [**detectron2**](https://github.com/facebookresearch/detectron2)

```
pip3 install -U torch torchvision cython
pip3 install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git3 clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip3 install -e detectron2_repo
```


### ZeroWaste COCO Dataset - Setup

The location of the ZeroWaste Azure Blob can be accessed here: _[TODO]:need to add link and locaiton and access instructions_

And make sure to put the files as the following structure:
```
zerowasteimages
├── dataset_config
|   ├── ds1_3class_test_coco_instances.json
│   ├── ds1_3class_test_mask_definitions.json
│   ├── ds1_3class_train_coco_instances.json
│   ├── ds1_3class_train_mask_definitions.json
│   ├── ...
|
└── images
    ├── imagedb
    ├── raw
    ├── masked
    ├── augmentations
    ├── ...
```
When training, change the root path to your own data path.


## Getting Started

```
git submodule update --init --recursive
```

pretrain model: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

### Get pretrained model
Download pretrained ResNet50 params from the following url.
```
mkdir pretrained_model
cd pretrained_model
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
mv resnet50-19c8e357.pth resnet50.pth
```
Get the pretrained RetinaNet by run the script:
```
cd network
python get_state_dict.py
```


## Training


### Training - Quickstart

### Training from scratch

TODO

### Training from pretrained checkpoint

For cases where we need to `resume` or `load_checkpoint` we use the following argmuent:

```
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1
```




### Troubleshooting

#### Mask RCNN Errors

Using the Mask_RCNN from https://github.com/matterport/Mask_RCNN produces `AttributeError: 'Model' object has no attribute 'metrics_tensors'`. This is fixes by updated the `mrcnn/model.py` file on line `2191`:

```py
  # Add metrics for losses
  for name in loss_names:
      self.keras_model.metrics_tensors = []   # You should add this code
      if name in self.keras_model.metrics_names:
          continue
      layer = self.keras_model.get_layer(name)
      self.keras_model.metrics_names.append(name)
      loss = (
          tf.reduce_mean(layer.output, keepdims=True)
          * self.config.LOSS_WEIGHTS.get(name, 1.))
      self.keras_model.metrics_tensors.append(loss)
```

---
### References

1. https://github.com/matterport/Mask_RCNN
2. https://github.com/dbolya/yolact#installation