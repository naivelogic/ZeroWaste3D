#custom_docker_image='yolact:1'
import sys, os, shutil

import argparse

from azureml.core import Run
# dataset object from the run
run = Run.get_context()
print(">>>>> RUN CONTEXT <<<<<<")
print(run)


print(f'\ndirectory listing (currnt): \n{os.listdir("./")}')
shutil.copy("./code/custom_yolact_config.py", "/yolact/data/config.py")
shutil.copy("./code/custom_yolact_trainer.py", "/yolact/train.py")
print("moved file")
print(f'\ndirectory listing (yolact): \n{os.listdir("/yolact/")}')





def get_parser():
    parser = argparse.ArgumentParser(description="ZeroWaste Detectron2 ML Training models")
    parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
    parser.add_argument('--img-folder', type=str, dest='img_folder', help='data folder mounting point')
    parser.add_argument('--masks-folder', type=str, dest='masks_folder', help='data folder mounting point')
    parser.add_argument('--output-folder', type=str, dest='output_folder', help='data folder mounting point')
    parser.add_argument('--config-file', type=str, dest='config_file', help='training configuraiton ad parameters')
    return parser


args = get_parser().parse_args()
DATA_FOLDER = args.data_folder
LOGS_AND_MODEL_DIR = args.output_folder

# Register Custom Dataset
MASKS_PATHS = args.masks_folder
IMG_PATHS = args.img_folder

print('WHAT ISN THE DATA CONFIG FOLDER??')
print(MASKS_PATHS)


TRAIN_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_train_coco_instances.json')
VAL_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_val_coco_instances.json')
TEST_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_test_coco_instances.json')

print(f'currenting ALM aguments before train defauls {args.__dict__}')

sys.path.append("/yolact/") # go to parent dir
from train import *


args.config=args.config_file 
set_cfg(args.config)

cfg.dataset.train_images = IMG_PATHS
cfg.dataset.train_info = TRAIN_PATH 
cfg.dataset.valid_images = IMG_PATHS    
cfg.dataset.valid_info = VAL_PATH  

cfg.backbone.path = os.path.join(DATA_FOLDER,'03-experiments/01-pretrained_models/yolact_plus_resnet50_54_800000.pth') 



args.log_folder=LOGS_AND_MODEL_DIR
args.save_folder=LOGS_AND_MODEL_DIR


print(args.__dict__)

print(f'current CFGs')
print(f'dataset name: {cfg.dataset.name}\ndatset training images: {cfg.dataset.train_images}')
print(f'databset backbonme: {cfg.backbone.__dict__}')
train()
    