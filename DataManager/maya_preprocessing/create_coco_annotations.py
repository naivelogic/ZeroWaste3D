import os, json, cv2, shutil
import numpy as np
from PIL import Image
import itertools 
import time
import maya_parseutils as sx
from tqdm import tqdm
from submask_utils import create_sub_masks, create_sub_mask_annotation

import sys
sys.path.append("../") 

#from dataset_configs.zerowaste_ds_config import *
from dataset_configs.waterwaste_ds_config import *


# These ids will be automatically increased as we go
annotation_id = 1
image_id = 1
is_crowd = 0
images=[]
annotations = []


def create_annotaitons(params):
    global annotation_id, image_id, is_crowd, images, annotations, CATEGORY_LIST

    data_path = os.path.join(params[0],params[1])

    or_img_name =data_path+'.jpg'
    or_mask_name = data_path+'.png'
    original_sub_mask_path = data_path+'.json'

    org_mask_cat_ids = json.load(open(original_sub_mask_path)) # Read data from file

    ## move and rename RBG Image
    rbg_base_name = "{:06d}.jpg".format(image_id)
    target_folder_rgb = os.path.join(params[2], 'images', rbg_base_name)

    shutil.copy(or_img_name, target_folder_rgb)

    ### CREATE MASK ANNOTATIONS
    mask_image = Image.open(or_mask_name)
    sub_masks = create_sub_masks(mask_image)

    for color, sub_mask in sub_masks.items():
        #print(color, category_idS[color])
        category_id = CATEGORY_IDS[org_mask_cat_ids[color]]
        annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
        annotations.append(annotation)
        annotation_id += 1

    img_size = mask_image.size    
    new_img={}
    new_img["license"] = 0
    new_img["file_name"] = rbg_base_name
    new_img["width"] = img_size[0]
    new_img["height"] = img_size[1]
    new_img["id"] = image_id
    images.append(new_img)


    image_id += 1


def get_tmp_dir_list(tmp_path):
    o = os.listdir(tmp_path)
    ol1 = [os.path.splitext(x)[0] for x in o]
    tmp_file_names = np.unique(ol1)
    return tmp_file_names


def main(inPath, outPathRoot):
    global annotation_id, image_id, is_crowd, images, annotations, CATEGORY_LIST
    
    data_files = get_tmp_dir_list(inPath)
    for file in tqdm(data_files):
        create_annotaitons([inPath, file, outPathRoot])

    print("processing done")

    print("saving annotations to coco as json ")
    ### create COCO JSON annotations
    my_dict = {}
    my_dict["info"]= COCO_INFO
    my_dict["licenses"]= COCO_LICENSES
    my_dict["images"]=images
    my_dict["categories"]=COCO_CATEGORIES
    my_dict["annotations"]=annotations

    # TODO: specify coco file locaiton 
    output_file_path = os.path.join(outPathRoot,"coco_instances.json")
    with open(output_file_path, 'w+') as json_file:
        json_file.write(json.dumps(my_dict))

    print(">> complete. find coco json here: ", output_file_path)
    print("last annotation id: ", annotation_id)
    print("last image_id: ", image_id)

if __name__ == "__main__":

    #inPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1_storm/raw/ds1.1/'
    #outPathRoot = '/home/redne/ZeroWaste3D/DataManager/coco_tools/outputs/ds/'
    #outPathRoot = '/home/redne/ZeroWaste3D/DataManager/coco_tools/outputs/ds/'
    #main(inPathRoot, outPathRoot, processing_multi_raw=False)

    # ds2 WW 11/14/20 - new multi raw (4k) - 2 maya runs
    inPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds2_storm/tmp/'
    outPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds2_storm/'
    
    if not os.path.exists(outPathRoot+'images'):
        os.mkdir(outPathRoot+'images')

    main(inPathRoot, outPathRoot)
