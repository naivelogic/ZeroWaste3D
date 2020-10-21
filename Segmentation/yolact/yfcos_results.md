

### For Yolact FCOS

| Model       | Backbone | Head      | Data | Date   | lr  | AP   | AP50 | AP75 | APs | APm  | APl  | Details           |
| ----------- | -------- | --------- | ---- | ------ | --- | ---- | ---- | ---- | --- | ---- | ---- | ----------------- |
| Yolact_fcos | R50-C4   | C5-512ROI | ds 2 | 102020 | 1X  | 57.7 | 72.9 | 63.2 | 5.2 | 56.7 | 83.8 | yfcos_ds2_r50_x00 |
| Yolact_fcos | R50-C4   | C5-512ROI | ds 1 | 061620 | 1X  | 15.3 | 34.1 | 9.16 | nan | 25.7 | 14.5 | TBD               |
| Yolact_fcos | R50-C4   | C5-128ROI | ds 2 | 062820 | 1X  | 7.42 | 21.5 | 1.61 | nan | 31.5 | 7.56 | TBD               |
| Yolact_fcos | TBD      | TBD       | ds 1 | 063020 | TB  | 18.2 | 24.5 | 10.2 | nan | 18.0 | 20.1 | yfcos_R50_ds1_x2  |
| TBD         |          |           |      |        |     |      |      |      |     |      |      | yfcos_ds2_r50_x08 |
| TBD         |          |           |      |        |     |      |      |      |     |      |      |                   |


#### Per Category Bbox MAP 

For each of hte pre-category the metrics can be read as `bbox AP` / `seg map AP`

| Experiment         | Chkpt | Dataset | Date   | utensils    | coffeeCup   | clearCup    |
| ------------------ | ----- | ------- | ------ | ----------- | ----------- | ----------- |
| **yfcos_ds2_r50_x00**   | 64k  | ds 2    | 102020 | 94.1 / 06.8 | 96.5 / 82.6 | 98.9 / 83.7 |
| yfcos_R50_ds1_x2   | TBD   | ds 1    | 061620 | 31.1 / 14.7 | 1.98 / 19.3 | 4.95 / 20.6 |
| yfcos_ds2_r50_x08  | 5k    | ds 2    | 063020 | 82.8 / 32.3 | 70.7 / 71.7 | 80.5 / 85.8 |
| yfcos_ds2_r101_x02 | 5k    | ds 2    | 063020 | 82.7 / 28.9 | 64.9 / 74.6 | 78.9 / 87.2 |
| yfcos_ds2_r50_x07  | 5k    | ds 2    | 062920 | NA / NA     | NA  / NA    | NA  / NA    |
| yfcos_ds2_r50_x06  | 4k    | ds 2    | 062920 | TBD / TBD   | TBD / TBD   | TBD / TBD   |
| TBD                | 5k    | ds 2    | 063020 | /           | /           | /           |

## Experimental Details

[ds2x01]: configuraiton lines and model downloads



#### yfcos_ds2_r50_x0 step5k
RUN ID: yolact_fcos_1593517754_6743927a
Z03

Total inference time: 0:01:11.266758 (0.838432 s / img per device, on 1 devices)
Total inference pure compute time: 0:01:05 (0.772987 s / img per device, on 1 devices)


MASK

|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
| :----: | :----: | :----: | :---: | :----: | :----: |
| 63.302 | 94.224 | 60.883 |  nan  | 57.892 | 65.292 |

BBOX 

|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
| :----: | :----: | :----: | :---: | :----: | :----: |
| 78.043 | 96.669 | 92.845 |  nan  | 78.996 | 77.876 |

[06/30 19:23:50] fvcore.common.checkpoint INFO: Loading checkpoint from /zerowastepublic/03-experiments/ds2/yolact_fcos/yfcos_ds2_r50_x08/model_0004999.pth
[06/30 19:23:55] d2.data.datasets.coco INFO: Loaded 90 images in COCO format from /zerowastepublic/02-datasets/ds2/dataset_config/ds2_3c_val_coco_instances.json


#### yfcos_ds2_r101_x02 step5k


RUN ID: yolact_fcos_1593518871_e226883d
z7

Loading checkpoint from /zerowastepublic/03-experiments/ds2/yolact_fcos/yfcos_ds2_r101_x02/model_0004999.pth
Loaded 90 images in COCO format from /zerowastepublic/02-datasets/ds2/dataset_config/ds2_3c_val_coco_instances.json
Total inference pure compute time: 0:01:19 (0.932798 s / img per device, on 1 devices)
Total inference time: 0:01:29.674724 (1.054997 s / img per device, on 1 devices)
Saving results to yfcos_ds2_r101_x02/inference/coco_instances_results.json

MASK

|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
| :----: | :----: | :----: | :---: | :----: | :----: |
| 63.573 | 94.653 | 63.150 |  nan  | 56.898 | 65.985 |

| category | AP     | category  | AP     | category | AP     |
| :------- | :----- | :-------- | :----- | :------- | :----- |
| utensils | 28.930 | coffeeCup | 74.566 | clearCup | 87.223 |


BBOX

|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
| :----: | :----: | :----: | :---: | :----: | :----: |
| 75.475 | 96.240 | 93.013 |  nan  | 76.044 | 75.756 |

| category | AP     | category  | AP     | category | AP     |
| :------- | :----- | :-------- | :----- | :------- | :----- |
| utensils | 82.652 | coffeeCup | 64.881 | clearCup | 78.891 |


[06/30 19:52:09 fvcore.common.checkpoint]: Loading checkpoint from /zerowastepublic/03-experiments/ds2/yolact_fcos/yfcos_ds2_r101_x02/model_0004999.pth
[06/30 19:52:19 d2.data.datasets.coco]: Loaded 90 images in COCO format from /zerowastepublic/02-datasets/ds2/dataset_config/ds2_3c_val_coco_instances.json



### For standard Faster RCNN

| Experiment | Backbone | Head | Data | Date | lr  | AP  | AP50 | AP75 | APs | APm | APl | Details           |
| ---------- | -------- | ---- | ---- | ---- | --- | --- | ---- | ---- | --- | --- | --- | ----------------- |
| TBD        |          |      |      |      |     |     |      |      |     |     |     | yfcos_ds2_r50_x08 |
| TBD        |          |      |      |      |     |     |      |      |     |     |     |                   |
