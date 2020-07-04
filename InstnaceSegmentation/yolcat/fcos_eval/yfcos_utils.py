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


import pickle
class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))
    
    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]: num_true += 1
            else: num_false += 1
            
            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)