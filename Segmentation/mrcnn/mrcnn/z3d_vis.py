from mrcnn import visualize
import os
import sys
import skimage
import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


class Vis(object):
    def __init__(self, class_name, model):
        self.class_names = class_name
        self.model = model

    def detect_and_visualize(self, img, img_arr):
        results = self.model.detect([img_arr], verbose=1)
        r = results[0]
        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                    self.class_names, r['scores'], figsize=(15, 10))#, save_vis = 'saved_img.jpg')

    def display_inference_folder(self, image_dir):
        image_paths = []
        for filename in os.listdir(image_dir):
            if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
                image_paths.append(os.path.join(image_dir, filename))

        for image_path in image_paths:
            #img = skimage.io.imread(image_path)
            img = cv2.imread(image_path)
            img_arr = np.array(img)
            self.detect_and_visualize(img, img_arr)

    def display_single_image(self, image_path):
        #img = cv2.imread('/home/redne/ZeroWaste3D/media/sample_images/3d_reconstruction_girl.png')
        #img = skimage.io.imread(image_path)
        img = cv2.imread(image_path)
        img_arr = np.array(img)
        self.detect_and_visualize(img, img_arr)

