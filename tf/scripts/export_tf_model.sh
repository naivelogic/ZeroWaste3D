#!/bin/bash
## tmux ls
## tmux [value] attach -t
## cat > export_model.sh
## sh export_model.sh


# EXPORT
mkdir /ZeroWaste/mnt/zerowaste_blob/project_zero/experiments/ds0_v1/ds0_v1_ssdlitev2_bc01_052420_step20k
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/ZeroWaste/experiments/ds0_v1_ssdlitev2_bc01.config
TRAINED_CKPT_PREFIX=/ZeroWaste/outputs/ds0_v1_ssdlitev2_bc01_052420/model.ckpt-20000
EXPORT_DIR=/ZeroWaste/mnt/zerowaste_blob/project_zero/experiments/ds0_v1/ds0_v1_ssdlitev2_bc01_052420_step20k
python training/models/research/object_detection/export_inference_graph.py --input_type=${INPUT_TYPE} --pipeline_config_path=${PIPELINE_CONFIG_PATH} --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} --output_directory=${EXPORT_DIR}

### test tf record for cf metrix
# update here
INFERENCE_GRAPH_PATH=/ZeroWaste/mnt/zerowaste_blob/project_zero/experiments/ds0_v1/ds0_v1_ssdlitev2_bc01_052420_step20k/frozen_inference_graph.pb
TFRECORD_PATH=/ZeroWaste/tfrecords/val.tfrecord
DETECTION_TFRECORD_PATH=/ZeroWaste/mnt/zerowaste_blob/models/evaluation/test_detection.tfrecord

python /ZeroWaste/training/models/research/object_detection/inference/infer_detections.py --input_tfrecord_paths=${TFRECORD_PATH} --output_tfrecord_path=${DETECTION_TFRECORD_PATH} --inference_graph=${INFERENCE_GRAPH_PATH}

### Create the confusion matrix

# update here
CONFUSION_MATRIX_PATH=/ZeroWaste/mnt/zerowaste_blob/project_zero/experiments/ds0_v1/ds0_v1_ssdlitev2_bc01_052420_step20k/confusion_matrix2.csv
DETECTION_TFRECORD_PATH=/ZeroWaste/mnt/zerowaste_blob/models/evaluation/test_detection.tfrecord
LABELMAP_PATH=/ZeroWaste/tfrecords/labelmap.pbtxt
python /ZeroWaste/mnt/zerowaste_blob/models/evaluation/confusion_matrix_dev.py --detections_record=${DETECTION_TFRECORD_PATH} --label_map=${LABELMAP_PATH} --output_path=${CONFUSION_MATRIX_PATH}

# update here
cat /ZeroWaste/mnt/zerowaste_blob/project_zero/experiments/ds0_v1/ds0_v1_ssdlitev2_bc01_052420_step20k/confusion_matrix2.csv