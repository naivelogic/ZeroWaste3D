
import argparse
import logging
import os, sys
import warnings
from collections import OrderedDict

import torch
from azureml.core import Run
from torch.nn.parallel import DistributedDataParallel

import detectron2
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_train_loader)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

warnings.filterwarnings("ignore", category=FutureWarning)

# dataset object from the run
run = Run.get_context()
print(">>>>> RUN CONTEXT <<<<<<")
print(run)

def get_parser():
    parser = argparse.ArgumentParser(description="ZeroWaste Detectron2 ML Training models")
    parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
    parser.add_argument('--img-folder', type=str, dest='img_folder', help='data folder mounting point')
    parser.add_argument('--masks-folder', type=str, dest='masks_folder', help='data folder mounting point')
    parser.add_argument('--output-folder', type=str, dest='output_folder', help='data folder mounting point')
    parser.add_argument('--config-file', type=str, dest='config_file', help='training configuraiton ad parameters')
    return parser

if __name__ == "__main__":
    print(f'\ndirectory listing (currnt): \n{os.listdir("./")}')

    args = get_parser().parse_args()
    DATA_FOLDER = args.data_folder
    
    
    TRAINING_CONFIG = args.config_file 
    TRAIN_EXPERIMENT_NAME = os.path.basename(TRAINING_CONFIG).split('.yaml')[0]
    LOGS_AND_MODEL_DIR = os.path.join(args.output_folder,TRAIN_EXPERIMENT_NAME) 

    # Register Custom Dataset
    MASKS_PATHS = args.masks_folder
    IMG_PATHS = args.img_folder

    print("#################################################")
    print(f'default training arguments before our custom training configs applied:')
    print(args.__dict__)
    print(f'Argument Summary')
    print(f'Data folder: {DATA_FOLDER}\nImage Folder: {IMG_PATHS}\nMask Folder: {MASKS_PATHS}')

    TRAIN_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_train_coco_instances.json')
    VAL_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_val_coco_instances.json')
    TEST_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_test_coco_instances.json')

    register_coco_instances(f"custom_dataset_train", {},TRAIN_PATH , IMG_PATHS)
    register_coco_instances(f"custom_dataset_val", {}, VAL_PATH, IMG_PATHS)
    register_coco_instances(f"custom_dataset_test", {}, TEST_PATH, IMG_PATHS)

    cfg = get_cfg()
    
    # check if we are training a tensormask
    if 'tensormask' in TRAINING_CONFIG:
        sys.path.append("/detectron2/projects/TensorMask/") # go to parent dir
        from tensormask import add_tensormask_config
        add_tensormask_config(cfg)

    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(TRAINING_CONFIG)
    #cfg.DATASETS.TRAIN = ("custom_dataset_train",)
    #cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    #cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  
    #cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    #cfg.SOLVER.IMS_PER_BATCH = 2
    
    #cfg.SOLVER.BASE_LR = 0.02 #0.002
    #cfg.SOLVER.MAX_ITER = 100    
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (utensiles, coffeeCup, clearCup)
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  

    #cfg.OUTPUT_DIR= LOGS_AND_MODEL_DIR

    #os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_AND_MODEL_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    print(f'Config all dump')
    print("")
    # https://detectron2.readthedocs.io/tutorials/configs.html?highlight=cfg
    print(cfg.dump())  # print formatted configs
    print("")
    print(f">>>>>> STARTING TRIANING for Experiment NAme - TBD <<<<<<<<")


    trainer.train()

        
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    #cfg.DATASETS.TEST = ("custom_dataset_val", )
    #predictor = DefaultPredictor(cfg)


    #test_metadata = MetadataCatalog.get("custom_dataset_test")
    #dataset_dicts = DatasetCatalog.get("custom_dataset_test")


    #print(test_metadata)



