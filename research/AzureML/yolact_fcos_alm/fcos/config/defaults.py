from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# FPN configs
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]

# fcos head configs
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.NUM_SHARED_CONVS = 0
_C.MODEL.FCOS.NUM_STACKED_CONVS = 4
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.USE_DEFORMABLE = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0

# Inference parameters
_C.MODEL.FCOS.SCORE_THRESH_TEST = 0.05
_C.MODEL.FCOS.NMS_THRESH_TEST = 0.6
_C.MODEL.FCOS.NMS_PRE_TOPK = 1000
_C.MODEL.FCOS.NMS_POST_TOPK = 100

# ---------------------------------------------------------------------------- #
# hyperparameters for Improvements part on Table 3 from fcos paper
# ---------------------------------------------------------------------------- #

# 1. apply GroupNormalization on fcos_head
_C.MODEL.FCOS.NORM = "none"  # Support "GN" or none

# 2. centerness branch on regression branch
_C.MODEL.FCOS.CTR_ON_REG = False

# 3. center sampling
_C.MODEL.FCOS.CENTER_SAMPLE = False
_C.MODEL.FCOS.POS_RADIUS = 1.5

# 4. IoU loss type. ['iou', 'linear_iou', 'giou']
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'iou'

# 5. Normalization of fcos regression targets by each level of fpn stride.
_C.MODEL.FCOS.NORMALIZE_REG_TARGETS = False
