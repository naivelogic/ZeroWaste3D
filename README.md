# ZeroWaste3D
Reducing Waste Contamination _(e.g., pepsi can in compost bin)_ with 3D Reconstruction using Photogrammetry. 


## Summary

We seek to generate synthetic datasets for training object reconition models to assist in the application of detecting and classifying object thrown in the trash as Compostable / Recyclable / Trash (Landfill).


### Problem Statement

Based on the work from the previous ZeroWaste 2018 and 2019 Hackathons __[TODO:]__ _(link to repos and blog post)_ that successfully addressed the use case of developing a DNN classifier to detect a variety of images, in the current events of COVID-19 that prevented new data and evaluations to improve the accuracy of the classifier, we seek to assess the ability and techniques to utilize 3D Reconstruction using photogrammetry to either enhance or independenly impove the accuracy by the improved approach in generating a new dataset. 


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

## Results

## Instance Segmentation (Summary)

Below are a highlight in key results from Instance Segmentation training. For more details please refer to [InstanceSegmentation](InstnaceSegmentation/Data-Processing.md)



| Model       | Backbone | Head      | Data | Date   | lr | AP   | AP50 | AP75 | APs  | APm  | APl  | 
| ----------- | -------- | --------- | ---- | ------ | -- | ---- | ---- | ---- | ---- | ---- | ---- |
| Yolact_fcos | R50-C4   | C5-512ROI | ds 1 | 061620 | 1X | 15.3 | 34.1 | 9.16 | nan  | 25.7 | 14.5 |
| Yolact_fcos | R50-C4   | C5-128ROI | ds 2 | 062820 | 1X | 7.42 | 21.5 | 1.61 | nan  | 31.5 | 7.56 |

