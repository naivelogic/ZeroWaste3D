import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances


TEST_PATH = '/mnt/zerowastepublic/02-datasets/ds2/dataset_config/ds2_3c_test_coco_instances.json'
register_coco_instances(f"custom_dataset_test", {}, TEST_PATH, IMG_PATHS)

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch, os
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = 'output/model_final.pth'
cfg.DATASETS.TEST = ("custom_dataset_test", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
predictor = DefaultPredictor(cfg)

from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
model = build_model(cfg)
evaluator = COCOEvaluator("custom_dataset_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "custom_dataset_test")
#a = inference_on_dataset(model, val_loader, evaluator)
inference_on_dataset(model, val_loader, evaluator)

#could pickle the results for later