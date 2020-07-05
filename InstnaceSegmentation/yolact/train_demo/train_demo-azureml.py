
import argparse
import logging
import os
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


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
    ap.add_argument('--img-folder', type=str, dest='img_folder', help='data folder mounting point')
    ap.add_argument('--masks-folder', type=str, dest='masks_folder', help='data folder mounting point')
    ap.add_argument('--config-file', type=str, dest='config_file', help='training configuraiton ad parameters')
    ap.add_argument('--num-gpus', type=int, default=1, dest='num_gpus', help='number of gpus *per machine')
    ap.add_argument("--num-machines", type=int, default=1,dest='num_machines', help="total number of machines")
    ap.add_argument("--opts",help="Modify config options using the command-line 'KEY VALUE' pairs",dest='opts',default=[],nargs=argparse.REMAINDER,)

    args = vars(ap.parse_args())
    print("#################################################")
    print("All Aguments: \n", args)

    DATA_FOLDER = args["data_folder"]
    IMG_PATHS = args["img_folder"]
    MASKS_PATHS = args["masks_folder"]
    TRAIN_CONFIG = args["config_file"]
    print("#################################################")
    print(f'Argument Summary')
    print(f'Data folder: {DATA_FOLDER}\nImage Folder: {IMG_PATHS}\nMask Folder: {MASKS_PATHS}\nTraining config yml: {TRAIN_CONFIG}')

    #MASKS_PATHS = os.path.join(os.path.abspath("."),data_folder, 'experiments/dataset_config')

    TRAIN_PATH = os.path.join(MASKS_PATHS, 'ds1_3class_train_coco_instances.json')
    VAL_PATH = os.path.join(MASKS_PATHS, 'ds1_3class_val_coco_instances.json')
    TEST_PATH = os.path.join(MASKS_PATHS, 'ds1_3class_test_coco_instances.json')

    register_coco_instances(f"custom_dataset_train", {},TRAIN_PATH , IMG_PATHS)
    register_coco_instances(f"custom_dataset_val", {}, VAL_PATH, IMG_PATHS)
    register_coco_instances(f"custom_dataset_test", {}, TEST_PATH, IMG_PATHS)

    

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

    cfg.OUTPUT_DIR= './output/zw_3class_x1'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

        
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("custom_dataset_val", )
    predictor = DefaultPredictor(cfg)


    test_metadata = MetadataCatalog.get("custom_dataset_test")
    dataset_dicts = DatasetCatalog.get("custom_dataset_test")


    print(test_metadata)



