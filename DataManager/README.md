# ZeroWaste3D Data Management

## Overview

TODO

## Data Preparation

TODO: summary of synthetics data generation (CAD models, backgrounds, rendering )
- CAD models
- backgrounds and textures
- rendering + lighting and camera variation
  
end deliverable: synthetics dataset for training object detection

## Quick Start

__Note: __ All datasets should be in linked to `[waterwaste_blob]/data` whose file system should be as following:
    ```
    ├── ds1
    │   ├── coco_ds
    │   │   ├── <scene_id>
    │   │   │   ├── rgb
    │   │   │   │   ├── <img_id>.png
    │   │   │   │   └── ...
    │   │   │   ├── mask_visib
    │   │   │   ├── depth
    │   │   │   └── ...
    ├── images
    ├── color_mask
    ├── raw
    │   ├── <camera_id><scene_id>
    │   │   ├── instance
    │   │   ├── rgba
    │   │   │   ├── <img_id>.png
    │   │   │   └── ...
    │   │   ├── depth
    │   └── ...
    ├── trash_ds
    │   ├── test/train_coco_instances.json
    │   ├── test/train.tfrecord
    │   ├── labelmap.pbtxt
    └── ...
    ```

1. create synthetics dataset in coco annotation
    ```
    python maya_preprocessing/create_coco.py /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1/raw/ /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1/
    ```

2. split coco dataset into train (80%) and test (20%) 

    ```
    python coco_tools/cocosplit.py --having-annotations -s 0.8 /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1/coco_instances.json /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1/coco_ds/train_coco_instances.json /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1/coco_ds/test_coco_instances.json
    ```

3. (optaion if using tfodapi detector) create tf_records

    ```
    TRAIN_IMAGE_DIR=/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1/images
    TEST_IMAGE_DIR=/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1/images
    TRAIN_ANNOTATIONS_FILE=/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1/coco_ds/train_coco_instances.json
    TEST_ANNOTATIONS_FILE=/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1/coco_ds/test_coco_instances.json
    OUTPUT_DIR=/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1/tf_ds/
    python coco_tools/create_coco_tf_record.py --logtostderr \
        --train_image_dir="${TRAIN_IMAGE_DIR}" \
        --test_image_dir="${TEST_IMAGE_DIR}" \
        --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
        --test_annotations_file="${TEST_ANNOTATIONS_FILE}" \
        --output_dir="${OUTPUT_DIR}"
    ```