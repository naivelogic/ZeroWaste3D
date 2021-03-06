"""
USAGE: For performing inferences on Yolcat FCOS we have to replace the original detectron2 installed `postprocessing.py` with this updated postprocessing file
as the original detectron2 file is only suit for ROI obtained masks. 
The path to be updated should be as such:

cat > ~/cv/lib/python3.6/site-packages/detectron2/modeling/postprocessing.py

-or-

/miniconda3/envs/py37/lib/python3.7/site-packages/detectron2/modeling/postprocessing.py

to run inferences then:
~/Yolact_fcos$ python ./train_net.py --config-file configs/Yolact/yolacat_fcos_R50_zwds1_3c.yaml --eval-only MODEL.WEIGHTS output/fcos/yolacat_fcos_R50_zwds1_3c/model_0029999.pth
"""

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.nn import functional as F

from detectron2.layers import paste_masks_in_image
from detectron2.structures import Instances


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.
    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.
    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]
    
    if results.has("pred_masks"):
        if results.pred_masks.shape[0]:
            results.pred_masks = F.interpolate(input=results.pred_masks, size=results.image_size,mode="bilinear", align_corners=False).gt(0.5).squeeze(1)
        #results.pred_masks = paste_masks_in_image(
        #    results.pred_masks[:, 0, :, :],  # N, 1, M, M
        #    results.pred_boxes,
        #    results.image_size,
        #    threshold=mask_threshold,
        #)

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.
    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.
    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.
    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result