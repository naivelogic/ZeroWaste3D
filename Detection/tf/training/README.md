
```
cat > mount_blob.sh
#!/bin/sh
sudo mkdir /mnt/blobfusetmp 
sudo chown $USER /mnt/blobfusetmp 
blobfuse ~/ZeroWaste/mnt/zerowaste_blob --tmp-path=/mnt/blobfusetmp  --config-file=/home/$USER/ZeroWaste/mnt/zerowaste_blob_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other 

sh mount_blob.sh

_https://msblox-03.visualstudio.com/_git/ZeroWaste?path=%2FMachineLearning%2Fexperiments%2Ffeb20_hack%2Ffeb20_hack_experiments.md&_a=preview_

sudo mkdir /mnt/blobfusetmp 
sudo chown $USER /mnt/blobfusetmp 
blobfuse ~/ZeroWaste/mnt/zerowaste_blob --tmp-path=/mnt/blobfusetmp  --config-file=/home/$USER/ZeroWaste/mnt/zerowaste_blob_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other 

cd ~/ZeroWaste/tfrecords
rm -R ./*
cp ~/ZeroWaste/mnt/zerowaste_blob/project_zero/experiments/ds01_v1/*.tfrecord ./
cp ~/ZeroWaste/mnt/zerowaste_blob/project_zero/experiments/ds01_v1/labelmap.pbtxt ./


tmux

nvidia-docker run -it -v /home/$USER/ZeroWaste:/ZeroWaste -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:1.12.0-gpu-py3 bash 

cd /ZeroWaste/training
echo "export PYTHONPATH=$(pwd)/models:$(pwd)/models/research:$(pwd)/models/research/slim:$(pwd)/cocoapi/PythonAPI" >> ~/.bashrc
cd models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd /ZeroWaste

python training/model_main.py --model_dir /ZeroWaste/outputs/ssd_incpetion_v2_core_exp_030120 --pipeline_config_path /ZeroWaste/experiments/ssd_incpetion_v2/pipeline.config 

```


----
#### Archive


```

#2 GPU DISTRIBUTED / PAREALLEL
NV_GPU=0 nvidia-docker run -it -v /home/$USER/ZeroWaste:/ZeroWaste -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:1.12.0-gpu-py3 bash

NV_GPU=1 nvidia-docker run -it -v /home/$USER/ZeroWaste:/ZeroWaste -p 8889:8888 tensorflow/tensorflow:1.12.0-gpu-py3 bash  

python training/model_main.py --model_dir /ZeroWaste/outputs/ssd_v2_q_core_exp_030120 --pipeline_config_path /ZeroWaste/experiments/ssd_v2_q/pipeline.config 


tensorboard --logdir overtrain=./ssdlite_v2_core_min_022520/,ssdlitev2_core_min=./ssdlite_v2_core_min_022920/ --window_title "ssdlite_v2_core_min SkynetAI v2" 


# EXPORT

mkdir /ZeroWaste/mnt/zerowaste_blob/models/experiments/feb20_hack/core/core_aug/ssd_v1_exp_aug_032320_step7k

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/ZeroWaste/experiments/ssd_v1/pipeline.config
TRAINED_CKPT_PREFIX=/ZeroWaste/outputs/ssd_v1_exp_aug_032320/model.ckpt-7866
EXPORT_DIR=/ZeroWaste/mnt/zerowaste_blob/models/experiments/feb20_hack/core/core_aug/ssd_v1_exp_aug_032320_step7k
python training/models/research/object_detection/export_inference_graph.py --input_type=${INPUT_TYPE} --pipeline_config_path=${PIPELINE_CONFIG_PATH} --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} --output_directory=${EXPORT_DIR}

### test tf record for cf metrix
TFRECORD_PATH=/ZeroWaste/tfrecords/val.tfrecord 
DETECTION_TFRECORD_PATH=/ZeroWaste/mnt/zerowaste_blob/models/evaluation/test_detection.tfrecord 
INFERENCE_GRAPH_PATH=/ZeroWaste/mnt/zerowaste_blob/models/experiments/feb20_hack/core/core_aug/ssd_v1_exp_aug_032320_step7k/frozen_inference_graph.pb

python /ZeroWaste/training/models/research/object_detection/inference/infer_detections.py --input_tfrecord_paths=${TFRECORD_PATH} --output_tfrecord_path=${DETECTION_TFRECORD_PATH} --inference_graph=${INFERENCE_GRAPH_PATH}


### Create the confusion matrix


LABELMAP_PATH=/ZeroWaste/tfrecords/labelmap.pbtxt
CONFUSION_MATRIX_PATH=/ZeroWaste/mnt/zerowaste_blob/models/experiments/feb20_hack/core/core_aug/ssd_v1_exp_aug_032320_step7k/confusion_matrix2.csv
DETECTION_TFRECORD_PATH=/ZeroWaste/mnt/zerowaste_blob/models/evaluation/test_detection.tfrecord

python /ZeroWaste/mnt/zerowaste_blob/models/evaluation/confusion_matrix_dev.py --detections_record=${DETECTION_TFRECORD_PATH} --label_map=${LABELMAP_PATH} --output_path=${CONFUSION_MATRIX_PATH}

cat /ZeroWaste/mnt/zerowaste_blob/models/experiments/feb20_hack/core/core_aug/ssd_v1_exp_aug_032320_step7k/confusion_matrix2.csv
```


Inference
- https://github.com/tensorflow/models/tree/master/research/object_detection