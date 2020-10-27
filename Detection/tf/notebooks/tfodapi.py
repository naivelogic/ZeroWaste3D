# based off of https://github.com/waterzhaojun/image_collect_tools
from distutils.version import StrictVersion
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

sys.path.append("/home/redne/repos/models/research/")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def load_model(MODEL_PATH_ROOT, MODEL_PATH, NUM_CLASSES):

    PATH_TO_CKPT = os.path.join(MODEL_PATH_ROOT, MODEL_PATH, "frozen_inference_graph.pb")
    PATH_TO_LABELS = os.path.join(MODEL_PATH_ROOT, MODEL_PATH,  "labelmap.pbtxt")

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return PATH_TO_CKPT, PATH_TO_LABELS, label_map, categories, category_index

def load_detection_graph(PATH_TO_FROZEN_GRAPH):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def run_inference_for_single_image(image, graph):
    # https://github.com/waterzhaojun/image_collect_tools/blob/master/crop_rodent.ipynb

    with graph.as_default():
        with tf.Session() as sess:    
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def crop_object_image(image_np_array, savepath, output_dict, normalized=True, save_cropped_img=True,
                      show_cropped_img=False, img_size=(12, 8), score_threshold=0.5):

    # get the boxes
    idxes = output_dict['detection_scores'] > score_threshold
    boxes = output_dict['detection_boxes'][idxes]

    # box is a list with size 4
    height, width, __ = image_np_array.shape

    for box in boxes:
        newbox = np.copy(box)
        if normalized:
            newbox = (newbox * [height, width, height, width]).astype(int)

        print(newbox)

        image_crop = image_np_array[newbox[0]:newbox[2], newbox[1]:newbox[3], :]

        if save_cropped_img:
            im = Image.fromarray(image_crop)
            im.save(savepath)

        if show_cropped_img:
            plt.figure(figsize=img_size)
            plt.imshow(image_crop)

