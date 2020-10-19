# ZeroWaste3D on Clean Ocean

Objective: Train accurate detector for detecting waste in ocean/waterways to aid in reduction of polution and waste contaimination. 


### Summary

1. Generate synthetic dataset of important objects using Maya
2. Train detector using Faster RCNN networks on generated synthetics dataset
3. Fine tune detector with real world data




### Results 

Performance of the training results is compared using the Mean Average Percisions scores. 

__WORK IN PROGRESS - Oct 2020__

| Train Dataset Images + Class info  | mAP  | mAPrecision IoU=0.5 | mARecall 100 |
|------------------------------------|------|---------------------|--------------|
| DS.X Synthetics (#XXK) - (#classes) | 0.XX | 0.XX                | 0.XX         |
| DS.X Synthetics (#XXK) - (#classes) | 0.XX | 0.XX                | 0.XX         |
| DS.X Synthetics (#XXK) + Real (XX)  | 0.XX | 0.XX                | 0.XX         |
| DS.X Real Images Only (#XXX)        | 0.XX | 0.XX                | 0.XX         |


### Dataset Geneartion

Syntheitics objects were generated by rendering the 3D models with varations in object positioning (e.g., object angles) and lighting conditions. 

* XX 3D models used in syntheitcs images generation pipline
* XXK images were synthetically generated


### Training Experiments

Dataset 0 training only utilized sythnetics dataset and only a selected number of categories to establish an initialbaseline. 

* 80/20 train/test split
* 