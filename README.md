# ZeroWaste3D
Reducing Waste Contamination _(e.g., pepsi can in compost bin)_ with 3D Reconstruction using Photogrammetry. 

![Image of Dataset2 Experiments Inferences June 2020](media/zerowaste_ds2_maskrcnn_valoutputs.gif#50x100)

## Summary

We seek to generate synthetic datasets for training object reconition models to assist in the application of detecting and classifying object thrown in the trash as Compostable / Recyclable / Trash (Landfill).


### Problem Statement

Based on the work from the previous ZeroWaste 2018 and 2019 Hackathons __[TODO:]__ _(link to repos and blog post)_ that successfully addressed the use case of developing a DNN classifier to detect a variety of images, in the current events of COVID-19 that prevented new data and evaluations to improve the accuracy of the classifier, we seek to assess the ability and techniques to utilize 3D Reconstruction using photogrammetry to either enhance or independenly impove the accuracy by the improved approach in generating a new dataset. 

## Applications
<table>
  <tr>
    <td>Waste Detection + Classification </td>
    <td>Pose Estimation (in Dev Winter 2020) </td>
  </tr>
  <tr>
    <td><img src=media/sample_images/ds2v1.png width=320 height=240></td>
    <td><img src=research/6D_Pose/sample/vis_gt_pose_coffeecup_yolo2_091420.jpg width=320 height=240></td>
  </tr>
</table> 

### Contents

+ `DataManager` - mainly coco data generation utilities and info on generated datasets. _(see `/research` for tools on synthetics)
  + `coco_manager.py` - generates dataset boxes and meshes dictionaries
  + `coco_json_utils.py` - loads labeled images and mask into a coco data structure format
  + TODO: on training augmentations scripts
+ `Detection` - object detector pipelines (e.g., pytorch, tensorflow)
+ `InstanceSegmentation` - mostly toolkits for fully convolutional modeels for realtime instance segmentation
+ `research` - various research topics including AzureML and Model Analysis details


### TODO: Summary and Performance of ZeroWaste 2019 Dataset

__TODO__

## Performance and Results

Performance evaluation from model training across the different detection models utilized the Average Percision (AP). For specific details please refer to [InstanceSegmentation Model Results](InstnaceSegmentation/MaskResults.md).

### Instance Segmentation (Summary)

Below are a highlight in key results from Instance Segmentation training. For more details please refer to 

| Backbone           | bbox | mask | RunID                                 | Model | Log_Results |
| ------------------ | ---- | ---- | ------------------------------------- | ----- | ----------- |
| yfcos_ds2_r50_x08  | 78.0 | 63.3 | z03 - yolact_fcos_1593517754_6743927a | TBD   | TBD         |
| yfcos_ds2_r101_x02 | 75.5 | 63.5 | z07 - yolact_fcos_1593518871_e226883d | TBD   | TBD         |
|                    |      |      |                                       |       |             |


## Quick Start

### Requirements and Installation

An `environment.yml` file is provided for installing the prerquists and dependencies for running various modules in this repository. 

### Training

Training the various netowrks can be ran locally/VM on a GPU within a Docker container or experiments deployed on AzureML. 

- To install yolact related models refer to the `InstanceSegmentation/` folder
- To install detectors models for pure object detection refer to the tensorflow and pytorch approaches in the `Detector/` folder
- For running in docker (yolact and detectron2) refer to `reserach/AzureML/Docker` folder
- For training on AzureML refer to the `research/AzureML/` folder