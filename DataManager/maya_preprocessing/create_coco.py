import os, json, cv2
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


def createColorMaskImage2(instanceImage, instanceLabels, categoryKey='CategoryPath',
                          categoryToColorMapping=CATEGORY_LIST):
    '''Creates a simplified color image for the instance image using masks
    @param instanceImage: Image with the instanceid
    @param instanceLabels: Result of reading the metadata file
    @param categoryKey: Instance metadata key
    @param categoryToColorMapping: dictionary of categoryvalue --> color
    @returns:Simplified color mapping masks'''
    w = instanceImage.shape[1]
    h = instanceImage.shape[0]
    colorMaskImage = np.zeros((h,w,3), np.uint8)
    uniqueImagePixelValues = np.unique(instanceImage)
    
    # Initialize a dictionary of sub-masks indexed by RGB colors
    category_ids = {}
    color_counter = 0
    for id in uniqueImagePixelValues:        
        if not id in instanceLabels.keys():
            continue
        if not categoryKey in instanceLabels[id].keys():
            continue

        ## UPDATE - need to remove pixels sizes too small < 20 (error found on px size 7 - mask not vis)
        ## removing the image and logging it 
        pixel_size = sx.getNumberOfPixels(instanceImage,id)
        if pixel_size <= 20:
            print(">> image_processing_error found when creating color mask image")
            return 'error', 'error'

        categoryValue = instanceLabels[id][categoryKey]
        if (not categoryValue in categoryToColorMapping):
            continue
        
        
        # for DS1.1 S_CUP and P_cup assingment (only needed for 11/10/20 but should consider adding fix as util)
        # also need to add other utils for adding/merging/fixing processes datasets (e.g., metadata for link back original raw file (just need InstnaceIDs))
        #if categoryValue == 'S_cup' and 'clear' in instanceLabels[id]['Name']:
        #    categoryValue = 'P_cup'

        color = sx.COLOR_CATEGORY[color_counter]
        
        colorMaskImage[instanceImage==id,0] = color[0]
        colorMaskImage[instanceImage==id,1] = color[1]
        colorMaskImage[instanceImage==id,2] = color[2]
        
        category_ids[str(color)] = categoryValue
        color_counter += 1
    return colorMaskImage, category_ids

def process_folder(params):
    global annotation_id, image_id, is_crowd, images, annotations, CATEGORY_LIST, cat_id

    #parses the results
    dataDirectory = params[0]
    outPathRoot = params[1]

    # parses the results
    results = sx.readDataDirectory(dataDirectory[2])
    if results == 'error':
        return

    #extracts the labeling metadata, rgb image, labeling image, depth
    instanceLabels = sx.parseInstancelabels(results['instance'][0])
    rgbImagePath = results['rgba'][1][0]
    instanceImagePath = results['instance'][1][0]

    # READ AND CREATE MASK (COLOR / BINARY)
    instanceImage = sx.readInstanceImage(instanceImagePath)
    outMaskImageForViewing, category_idS = createColorMaskImage2(instanceImage, instanceLabels)
    if category_idS == 'error':
        print("[removing file] >> mask processing error found in: ", rgbImagePath)
        return
    #print("mask cat color ids: ", category_idS)

    # READ AND CREATE RGB Image
    rgbImage = sx.readRgbImage(rgbImagePath)
    rgb_path = outPathRoot+"/images/{:06d}.jpg".format(image_id) # e.g. 00000023.jpg
    sx.saveImage(rgbImage, rgb_path)

    mask_path = outPathRoot+"/color_mask/{:06d}.png".format(image_id) # e.g. 00000023.jpg
    sx.saveImage(outMaskImageForViewing, mask_path)

    ### CREATE MASK ANNOTATIONS
    mask_image = Image.open(mask_path)
    sub_masks = create_sub_masks(mask_image)

    for color, sub_mask in sub_masks.items():
        #print(color, category_idS[color])
        category_id = CATEGORY_IDS[category_idS[color]]
        annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
        annotations.append(annotation)
        annotation_id += 1

    img_size = mask_image.size    
    new_img={}
    new_img["license"] = 0
    new_img["file_name"] = os.path.basename(rgb_path)
    new_img["width"] = img_size[0]
    new_img["height"] = img_size[1]
    new_img["id"] = image_id
    images.append(new_img)


    image_id += 1


def get_raw_subdirs(inPath):
    """
    returns a list of subdirs for processing multiple raw maya dirs (e.g., ran multiple raw renderings for the same dataset)
    example: '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds2_storm/raw/'
    """
    import glob
    raw_paths = glob.glob(os.path.join(inPath, '*', '*'), recursive=True)
    print("processing: {0} of raw maya files".format(len(raw_paths)))
    
    dataDirectories = []
    for raw_path in raw_paths:
        dataDirectories.extend(sx.listDataDirectories(raw_path))

    return dataDirectories

def main(inPath, outPathRoot, processing_multi_raw=False):
    global annotation_id, image_id, is_crowd, images, annotations, CATEGORY_LIST, cat_id
    
    if not os.path.exists(outPathRoot):
        os.mkdir(outPathRoot)
    if not os.path.exists(outPathRoot+'/images'):
        os.mkdir(outPathRoot+'/images')
    if not os.path.exists(outPathRoot+'/color_mask'):
        os.mkdir(outPathRoot+'/color_mask')
    
    if processing_multi_raw == True: 
        print("looking up multi raw dirs for processing...")
        dataDirectories = get_raw_subdirs(inPath)
    else:
        dataDirectories = sx.listDataDirectories(inPath)


    for dataDirectory in tqdm(dataDirectories):
        process_folder([dataDirectory, outPathRoot])

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


# if run from command line it parsers the parameters and runs main function
if __name__ == "__main__":
    #if (len(sys.argv) != 3):
    #    print('Usage: run.py <IN> <OUT>')
    #    sys.exit(0)
    #inPathRoot = sys.argv[1]
    #outPathRoot = sys.argv[2]

    # ds1 CSIRO  11/7/20
    #inPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1/raw/ds1.0/9d15240302634bb99c11b4d275d410cd-ndsina8hov12cdqbsh6qg0gf04'
    #outPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1_overlake'
    #inPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1_storm/raw/ds1.1/'
    #outPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1_storm'
    #main(inPathRoot, outPathRoot, processing_multi_raw=False)

    # ds2 WW 11/14/20 - new multi raw (4k) - 2 maya runs
    inPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds2_storm/raw/'
    outPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds2_storm'
    main(inPathRoot, outPathRoot, processing_multi_raw=True)

