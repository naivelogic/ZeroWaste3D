import sys
sys.path.append("../") 
from mrcnn.config import Config

class ZeroWaste_maskrcnn_config(Config):
    NAME = "waterwaste_ds0"
    BACKBONE = "resnet50"
    LEARNING_RATE = 0.001

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BATCH_SIZE=1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 5 (ds1)
    STEPS_PER_EPOCH = 500
    

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 