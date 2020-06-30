|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
| :----: | :----: | :----: | :---: | :----: | :----: |
| 12.674 | 24.509 | 10.221 |  nan  | 41.089 | 11.845 |



### For standard Faster RCNN

| Backbone |  TSD  |  AP   | AP_0.5 | AP_0.75 | AP_s  | AP_m  | AP_l  | Download |
| :------: | :---: | :---: | :----: | :-----: | :---: | :---: | :---: | :------: |
| ResNet50 |       | 36.2  |  58.1  |  39.0   | 21.8  | 39.9  | 46.1  |          |


| Model       | Backbone | Head      | Data | Date   | lr  | AP   | AP50 | AP75 | APs | APm  | APl  | Details          |
| ----------- | -------- | --------- | ---- | ------ | --- | ---- | ---- | ---- | --- | ---- | ---- | ---------------- |
| Yolact_fcos | R50-C4   | C5-512ROI | ds 1 | 061620 | 1X  | 15.3 | 34.1 | 9.16 | nan | 25.7 | 14.5 | TBD              |
| Yolact_fcos | R50-C4   | C5-128ROI | ds 2 | 062820 | 1X  | 7.42 | 21.5 | 1.61 | nan | 31.5 | 7.56 | TBD              |
| Yolact_fcos | TBD      | TBD       | ds 1 | 063020 | TB  | 18.2 | 24.5 | 10.2 | nan | 18.0 | 20.1 | yfcos_R50_ds1_x2 |
| TBD         |          |           |      |        |     |      |      |      |     |      |      |                  |
| TBD         |          |           |      |        |     |      |      |      |     |      |      |                  |


#### Per Category Bbox MAP 

For each of hte pre-category the metrics can be read as `bbox AP` / `seg map AP`

| Experiment       | Backbone | Dataset | Date   | utensils    | coffeeCup   | clearCup    |
| ---------------- | -------- | ------- | ------ | ----------- | ----------- | ----------- |
| yfcos_R50_ds1_x2 | R50-C4   | ds 1    | 061620 | 31.1 / 14.7 | 1.98 / 19.3 | 4.95 / 20.6 |



## Experimental Details

[ds2x01]: configuraiton lines and model downloads