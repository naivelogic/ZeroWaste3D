


## Data Processing
### Data Organization 

https://github.com/Flowingsun007/DeepLearningTutorial/tree/master/ObjectDetection/SSD/ssd-tf2

1. Data is located on share drive.
2. Unzip the file and place it in the 'dataset' folder, make sure the directory is like this : 
```
|——dataset
    |——VOCdevkit
        |——VOC2012
            |——Annotations
            |——ImageSets
            |——JPEGImages
            |——SegmentationClass
            |——SegmentationObject

```


## Parameters


## Parameters

| env | type | description |
| --- | --- | --- |
| BATCH_SIZE | int | Batch size. Default `32`. |
| EPOCHS | int | Epoch number. This template applies "Early stopping". Default `50`. |
| IMG_SIZE | int | Image size. **Currently only SSD300 (size=300) is supported!!** Automatically resize to this size. Default `300`. |
| SHUFFLE | bool | Shuffle train dataset. Default `true`. |
| RANDOM_SEED | int | Random seed. If set, use it for a data shuffling. Default `None`. |
| MAX_ITEMS | bool | Number of items to use. Default `None` which means use all items |
| TEST_SIZE | float | Ratio of test dataset. Default `0.4` |
| LEARNING_RATE | float | Learning rate. Need to be from `0.0` to `1.0`. Default `0.001`. |
| MOMENTUM | float | Momentum factor. Need to be from `0.0` to `1.0`. Default `0.0`. |
| WEIGHT_DECAY | float | Weight decay (L2 penalty). Need to be from `0.0` to `1.0`. Default `0.0`. |
| DAMPENING | float | Dampening for momentum. Need to be from `0.0`. Default `0.0`. |
| NESTEROV | float | Enables Nesterov momentum. Default `False`. |
| CONF_THRESHOLD | float | Confidence threshold to filter out bounding boxes with low confidence. Default `0.01`. |
| TOP_K | int | Number of bounding boxes to be taken. Default `200`. |
| NMS_THRESHOLD | float | The threshold for IoU to consider bounding boxes as the same. Default `0.45`. |
| OVERLAP_THRESHOLD | float | The overlap threshold used when matching boxes. Need to be from `0.0` to `1.0`. Default `0.5` |
| NEG_POS | int | Hard Negative Mining ratio. Default `3` |
| CONFIDENCE_THRESHOLD | float | Results above this threshold will be returned. **This option is valid only for prediction (or inference).** Default `0.1` |


---
## Resource
- https://github.com/pytorch/pytorch