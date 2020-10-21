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


## Performance and Results

_work in progress on sumerizing performance of ZeroWaste 2020 Synthetics Dataset

Performance evaluation from model training across the different detection models utilized the Average Percision (AP). For specific details please refer to [InstanceSegmentation Model Results](Segmentation/README.md).

### Instance Segmentation (Summary)

Below are a highlight in key results from Instance Segmentation training. We report mean average percision (mAP) at 0.5 IOU. For more details please refer to Segmentation directory.  

| Backbone           | Date  |bbox | mask | Model | Log_Results |
| ------------------ | ---- | ---- | ----- | ----------- | 
| yfcos_ds2_r50_x00  | 98.0 | 73.0 | Model link - TBD   | [logs](Segmentation/yolact/tools/yolact-fcos-eval/log/yfcos_ds2_r50_x00_102120.txt)         |


| Experiment         | Chkpt | Dataset | Date   | bbox | mask | Model Link  | Log Results |
| ------------------ | ----- | ------- | ------ | ---- | ---- | ----------- | ----------- |
| yfcos_ds2_r50_x00  | 64k   | ds 2    | 102020 | 98.0 | 73.0 | TBD         | [yfcos_ds2_r50_x00][1]|


#### Accuracy Performance Per Category Bbox mAP 

For each of hte pre-category the metrics can be read as `bbox AP` / `seg map AP`

| Experiment         | utensils    | coffeeCup   | clearCup    |
| ------------------ | ----------- | ----------- | ----------- |
| yfcos_ds2_r50_x00  | 94.1 / 06.8 | 96.5 / 82.6 | 98.9 / 83.7 |

---

## Quick Start

### Requirements and Installation

An `environment.yml` file is provided for installing the prerquists and dependencies for running various modules in this repository. 

### Training

Training the various netowrks can be ran locally/VM on a GPU within a Docker container or experiments deployed on AzureML. 

- To install yolact related models refer to the `InstanceSegmentation/` folder
- To install detectors models for pure object detection refer to the tensorflow and pytorch approaches in the `Detector/` folder
- For running in docker (yolact and detectron2) refer to `reserach/AzureML/Docker` folder
- For training on AzureML refer to the `research/AzureML/` folder


---
[1]: Segmentation/yolact/tools/yolact-fcos-eval/log/yfcos_ds2_r50_x00_102120.txt