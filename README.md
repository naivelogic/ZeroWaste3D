# ZeroWaste3D
Reducing Waste Contamination _(e.g., pepsi can in compost bin)_ with 3D Reconstruction using Photogrammetry. 

![](media/sample_images/3d_reconstruction_girl.png)

## Summary

We seek to generate synthetic datasets for training object reconition models to assist in the application of detecting and classifying object thrown in the trash as Compostable / Recyclable / Trash (Landfill).


### Problem Statement

Based on the work from the previous ZeroWaste 2018 and 2019 Hackathons __[TODO:]__ _(link to repos and blog post)_ that successfully addressed the use case of developing a DNN classifier to detect a variety of images, in the current events of COVID-19 that prevented new data and evaluations to improve the accuracy of the classifier, we seek to assess the ability and techniques to utilize 3D Reconstruction using photogrammetry to either enhance or independenly impove the accuracy by the improved approach in generating a new dataset. 


### TODO: Summary and Performance of ZeroWaste 2019 Dataset

__TODO__


### Results
|      Experiment     |               Model               | mAP@0.5      (accuracy) |   Loss   | Iteration |                 Model Name                |
|:-------------------:|:---------------------------------:|:-----------------------:|:--------:|:---------:|:-----------------------------------------:|
| utensils combined   | ssd mobilenet   v1 FPN            | 0.975                   | 0.488    | 6336      | ds0_v1_ssd_fpn_bc01_052420_step6k         |
| utensils   combined | ssdlite v2                        | 0.924                   | 0.999    | 20000     | ds0_v1_ssdlitev2_bc01_052420_step20k      |
| utensils combined   | faster rcnn   inception resnet v2 | 0.934                   | Missing  | 8468      | ds0_v1_frcnn_iresnetv2_baselinec01_052420 |
| baseline   - ds 1   | ssdlite v2                        | 0.892                   | 1.058    | 20000     | ds1v0_ssdlitev2_060220_step20k            |
| baseline - ds 1     | frcnn resnet50                    | 0.933                   | 0.366    | 50000     | ds1v0_frcnn_resnet50_060220_step50k       |
| baseline   - ds 1   | faster rcnn inception resnet v2   | 0.960                   | 0.432    | 50000     | ds1v0_frcnn_iresnetv2_060220_step50k      |
| baseline - ds 1     | faster rcnn   inception v2        | 0.880                   | 0.503    | 50000     | ds1v0_frcnn_incpetionv2_060220_step50k    |
