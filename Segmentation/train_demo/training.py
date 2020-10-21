import matplotlib.pyplot as plt
import numpy as np
import cv2


# detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

# register dataset
IMG_PATHS = '/mnt/zerowastepublic/02-datasets/ds2/images/'
TRAIN_PATH = '/mnt/zerowastepublic/02-datasets/ds2/dataset_config/ds2_3c_train_coco_instances.json'
VAL_PATH = '/mnt/zerowastepublic/02-datasets/ds2/dataset_config/ds2_3c_val_coco_instances.json'
register_coco_instances(f"custom_dataset_train", {},TRAIN_PATH , IMG_PATHS)
register_coco_instances(f"custom_dataset_val", {}, VAL_PATH, IMG_PATHS)

#metadataset = MetadataCatalog.get("custom_dataset_train")
#dataset_dicts = DatasetCatalog.get("custom_dataset_train")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import torch, os, sys

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("custom_dataset_train",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02 #0.002
cfg.SOLVER.MAX_ITER = 300    
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (utensiles, coffeeCup, clearCup)

# initialize model from model zoo
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  
#cfg.MODEL.WEIGHTS = 'output/model_final.pth' 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.SOLVER.BASE_LR = 0.00025

#cfg.SOLVER.MAX_ITER = 10000    
#cfg.SOLVER.CHECKPOINT_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()