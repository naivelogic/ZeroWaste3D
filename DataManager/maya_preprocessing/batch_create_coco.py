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

from multiprocessing import Pool
import uuid
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


    #parses the results
    dataDirectory = params[0]
    outPathRoot = params[1]

    # parses the results
    # parses the results
    results = sx.readDataDirectory(dataDirectory[2])
    if results == 'error':
        print("error found in: ", dataDirectory[2])
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
    tmp_id = str(uuid.uuid4())
    save_path = os.path.join(outPathRoot, tmp_id)
    rgbImage = sx.readRgbImage(rgbImagePath)
    rgb_path = save_path+".jpg" #outPathRoot+"/images/{:06d}.jpg".format(image_id) # e.g. 00000023.jpg
    sx.saveImage(rgbImage, rgb_path)

    mask_path = save_path+".png" #outPathRoot+"/color_mask/{:06d}.png".format(image_id) # e.g. 00000023.jpg
    sx.saveImage(outMaskImageForViewing, mask_path)

    temp_ann_path = save_path+".json"
    json.dump(category_idS,open( temp_ann_path, 'w' ))

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
    """
    print("Testing Debug")
    d = []
    d.extend(dataDirectories[15:23])
    d.extend(dataDirectories[2890:2899])
    d.extend(dataDirectories[1890:1899])
    d.extend(dataDirectories[890:899])
    d.extend(dataDirectories[3890:3899])
    return d
    """
    return dataDirectories


def main(inPath, outPathRoot, processing_multi_raw=False):

    if processing_multi_raw == True: 
        print("looking up multi raw dirs for processing...")
        dataDirectories = get_raw_subdirs(inPath)
    else:
        dataDirectories = sx.listDataDirectories(inPath)

    with Pool(processes=12) as pool:
        taskInputs = list([[dataDirectory, outPathRoot] for dataDirectory in dataDirectories])
        pool.map(process_folder, taskInputs)

    print("done")

if __name__ == "__main__":

    #inPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1_storm/raw/ds1.1/'
    #outPathRoot = '/home/redne/ZeroWaste3D/DataManager/coco_tools/outputs/ds/'
    #outPathRoot = '/home/redne/ZeroWaste3D/DataManager/coco_tools/outputs/ds/'
    #main(inPathRoot, outPathRoot, processing_multi_raw=False)

    # ds2 WW 11/14/20 - new multi raw (4k) - 2 maya runs
    inPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds2_storm/raw/'
    outPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds2_storm/tmp/'
    main(inPathRoot, outPathRoot, processing_multi_raw=True)
