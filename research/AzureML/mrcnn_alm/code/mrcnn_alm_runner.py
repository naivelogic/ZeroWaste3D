# USAGE with AZREML
# python alm_trainer.py --mode train


# import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils, visualize
import os
import argparse
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from azureml.core import Run


# https://github.com/MercyPrasanna/maskrcnn_segmentation/blob/master/segmentation/mask_rcnn/lesions.py

# dataset object from the run
run = Run.get_context()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
#ap.add_argument("-w", "--weights", help="optional path to pretrained weights")
#ap.add_argument("-m", "--mode", help="train or investigate")
args = vars(ap.parse_args())

data_folder = args["data_folder"]
DATA_FOLDER = data_folder
print('Data folder:', DATA_FOLDER)

print(">>> DATA CONFIGS >>>")
print(os.listdir(DATA_FOLDER))

# get the file paths on the compute
# '/home/redne/mnt/project_zero/project_zero/ds1/experiments/dataset_config/train_coco_instances.json'
# '/home/redne/mnt/project_zero/project_zero/ds1/parsed'
#IMAGES_PATHS = os.path.join(os.path.abspath("."),data_folder, 'project_zero/ds1/parsed')
#MASKS_PATHS = os.path.join(os.path.abspath("."),data_folder, 'project_zero/ds1/experiments/dataset_config')

#import yaml
#from yacs.config import CfgNode as new_cfg


#ds_cfg_dict = yaml.load(open("./DS_CONFIG.yaml"))
#ds_cfg = new_cfg(ds_cfg_dict)
#print("below is the DS_CONFIG from yacs")
#print(ds_cfg)


# Register Custom Dataset
#MASKS_PATHS = os.path.join(DATA_FOLDER, ds_cfg.FOLDERS.DATASET_CONFIG_FOLDER)
#IMG_PATHS = os.path.join(DATA_FOLDER, ds_cfg.FOLDERS.IMAGE_FOLDER)

#TRAIN_PATH = os.path.join(MASKS_PATHS, ds_cfg.COCO_DATASET.TRAIN)
#VAL_PATH = os.path.join(MASKS_PATHS, ds_cfg.COCO_DATASET.VAL)
#TEST_PATH = os.path.join(MASKS_PATHS, ds_cfg.COCO_DATASET.TEST)


MASKS_PATHS = os.path.join(DATA_FOLDER, '02-datasets/ds2/dataset_config/')
print(">>> DATA CONFIGS >>>")
print(MASKS_PATHS)
print(os.listdir(MASKS_PATHS))
print(">>> END DATA CONFIGS >>>")
IMG_PATHS = os.path.join(DATA_FOLDER, '02-datasets/ds2/images/')

TRAIN_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_train_coco_instances.json')
VAL_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_val_coco_instances.json')
TEST_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_test_coco_instances.json')

print("#################################################")
print(IMG_PATHS)
print("#################################################")
print(MASKS_PATHS)
print(f'Train COCO INSTNACE PATH: {TRAIN_PATH}')


# initialize the path to the Mask R-CNN pre-trained on COCO
COCO_PATH = "mask_rcnn_coco.h5"

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_PATH):
    utils.download_trained_weights(COCO_PATH)

# initialize the name of the directory where logs and output model
# snapshots will be stored

LOGS_AND_MODEL_DIR = "./outputs/lesions_logs"
os.makedirs(LOGS_AND_MODEL_DIR, exist_ok=True)




# Configuration
class x2_BaseConfig(Config):
    NAME = "maskrcnn_ds2_x3"
    BACKBONE = "resnet50"
    LEARNING_RATE = 0.001
    GPU_COUNT = 1#2
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 5 (ds1)
    STEPS_PER_EPOCH = 171
    BATCH_SIZE=1

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5
    

config = x2_BaseConfig()
config.display()


# Define the dataset
class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids




# Create the Training and Validation Datasets
dataset_train = CocoLikeDataset()
dataset_train.load_data(TRAIN_PATH, IMG_PATHS)
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data(VAL_PATH, IMG_PATHS)
dataset_val.prepare()

#Create the Training Model and Train

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOGS_AND_MODEL_DIR)
model.load_weights(COCO_PATH, by_name=True,exclude=["mrcnn_class_logits","mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])



print('>>> TRAINING STARTING')
print('>>> TRAINING: >> (1/2) TRAINING THE HEAD BREACHES')
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
start_train = time.time()
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=4, 
            layers='heads')
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')

print('>>> TRAINING: >> (1/2) TRAINING THE HEAD BREACHES >> COMPLETE!!')

print('>>> PT. 2 TRAINING STARTING')

print('>>> TRAINING: >> (2/2) FINE TUNE ALL LAYERS')
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
start_train = time.time()
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=8,  
            layers="all")
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')

print('>>> TRAINING: >> (2/2) FINE TUNE ALL LAYERS >> COMPLETE!!!')
print("... begin inferening")

