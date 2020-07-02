import sys, os
os.chdir("../Yolact_fcos/")
# import some common libraries
import json, torch, random, cv2
import matplotlib.pyplot as plt
import numpy as np


# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
# Register Custom Dataset
#BASE_MOUNT = '/home/redne/mnt/project_zero/project_zero/ds1/'
#MASKS_PATHS = os.path.join(BASE_MOUNT, 'experiments/dataset_config')
#IMG_PATHS = os.path.join(BASE_MOUNT, 'parsed')

MASKS_PATHS = '/mnt/zerowastepublic/02-datasets/ds2/dataset_config'
IMG_PATHS = '/mnt/zerowastepublic/02-datasets/ds2/images'

def register_datasets():
        
    TRAIN_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_train_coco_instances.json')
    VAL_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_val_coco_instances.json')
    TEST_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_test_coco_instances.json')

    register_coco_instances(f"custom_dataset_train", {},TRAIN_PATH , IMG_PATHS)
    register_coco_instances(f"custom_dataset_val", {}, VAL_PATH, IMG_PATHS)
    register_coco_instances(f"custom_dataset_test", {}, TEST_PATH, IMG_PATHS)

    ds_metadata = MetadataCatalog.get("custom_dataset_test")
    ds_dicts = DatasetCatalog.get("custom_dataset_test")

    print(f'\nMetadata Catalog from custom dataset:\n{ds_metadata}')

    return ds_metadata, ds_dicts



# SET UP MODEL

from detectron2.engine import DefaultPredictor
from fcos.config import get_cfg

#mymodel = 'yfcos_ds2_r101_x02'
mymodel = 'yfcos_ds2_r50_x08'
yfmnt = f'/mnt/zerowastepublic/03-experiments/ds2/yolact_fcos/{mymodel}/'
MODLE_PATH = os.path.join(yfmnt, 'model_final.pth')
MODLE_CFG_PATH = os.path.join(yfmnt, 'config.yaml')
METRICS_FILE = os.path.join(yfmnt, 'metrics.json')

def setup_model(mymodel, MODEL_PATH):
    cfg = get_cfg()
    cfg.merge_from_file(f"configs/Yolact/{mymodel}.yaml")
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = MODLE_PATH
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model, default was 0.7
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.6 # 0.3 for vids
    #cfg = cfg.merge_from_file(MODLE_CFG_PATH)
    print(f'MODEL WEIGHTS: {cfg.MODEL.WEIGHTS}\n')
    print(f'MODEL HEADS\n\n{cfg.MODEL.ROI_HEADS}\n\n')
    print(f'MODEL DATASETS\n\n{cfg.DATASETS}')
    predictor = DefaultPredictor(cfg)

    return cfg, predictor