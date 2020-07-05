
import random as r
import math
import cv2

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# detector threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
cfg.MODEL.WEIGHTS = 'output/model_final.pth'
cfg.DATASETS.TEST = ("custom_dataset_val", )
predictor = DefaultPredictor(cfg)

# get images with glob function to filelist to iterate through.  
# Preferred format of images: .png, .jpg, .jpeg.

filelist = glob.glob('*.jpg')

for i in range(10):
    img = cv2.imread(filelist[i])
    
    # prediction
    outputs = predictor(img)
    
    ### In case you want to see the detector visualizations uncomment the below script 
    
    ### v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    ### v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    ### cv2_imshow(v.get_image()[:, :, ::-1])
    
    # getting prediction bboxes from model outputs
    
    boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()[0]
    x2 = math.ceil(boxes[0])
    x1 = math.ceil(boxes[1])
    y2 = math.ceil(boxes[2])
    y1 = math.ceil(boxes[3])
    crop_img = img[x1:y1,x2:y2]
    #crop_img = cv2.resize(crop_img, (500,250))
    
    # showing original image
    cv2_imshow(img)
    
    # showing cropped of the object
    cv2_imshow(crop_img)