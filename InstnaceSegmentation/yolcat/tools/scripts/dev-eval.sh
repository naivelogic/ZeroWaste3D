#!/usr/bin/env bash

set -e

# Activate pytorch environment
source activate pytorch_p36

model_name=$1
if [ ! "$model_name" ]; then
    echo "Model name must be passed in as parameter"
    exit 1
fi

# Set up inference output dir
# ~/inference_output/ is used by detectron2 and will be logged to s3 after inference
# ~/output/ is used as persistent data store between different conductor commands; lives locally on the instance and not used directly by detectron2
rm -rf ~/inference_output/ || true
mkdir -p ~/inference_output/
cp -R ~/configs/ ~/inference_output/configs/
cp -R ~/output/training/$model_name/ ~/inference_output/weights/

# TEST! (use training in eval mode)
PYTHONPATH="/home/ubuntu/scripts:$PYTHONPATH"
python scripts/train_net.py --eval-only --config-file configs/${model_name}_test.yaml --num-gpus 4

# Copy final outputs
rm -rf ~/output/inference/$model_name/
mkdir -p ~/output/inference/$model_name/
cp ~/inference_output/inference/coco_instances_results.json ~/output/inference/$model_name/inferences.json

# tar and remove outputs dir
tar cvzf ~/outputs.tgz ~/inference_output/
ts=$(date +"%Y_%m_%d_%H_%M_%S")
aws s3 cp ~/outputs.tgz s3://maxar-analytics-data/transfer-car-detection/model_logs/inference/${model_name}_${ts}.tgz