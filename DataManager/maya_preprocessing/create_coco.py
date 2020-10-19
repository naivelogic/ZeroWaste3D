import os, json, cv2
import numpy as np
from PIL import Image
import itertools 
import time
import maya_parseutils as sx

from submask_utils import create_sub_masks, create_sub_mask_annotation

cat_id = {
    'H_beveragebottle': 1,
    'D_lid': 2,
    'S_cup': 3
}

CATEGORIES = ["H_beveragebottle", "D_lid", "S_cup"]

color_cat = [
    (0, 0, 255),          #//Blue
    (255, 0, 0),      #//Red
    (0, 255, 0),      #//Green
    (255, 0, 255),    #//Magenta
    (255, 128, 128),  #//Pink
    (128, 128, 128),  #//Gray
    (128, 0, 0),      #//Brown
    (255, 128, 0),    #//Orange
    (0, 255, 255)
]
#(255, 255, 0),    #//Yellow

# These ids will be automatically increased as we go
annotation_id = 1
image_id = 1
is_crowd = 0
images=[]
annotations = []

info = {
      "description":"CSIRO x MSFT Synthetics Water Waste Project - Dataset 0",
      "url":"",
      "version":"1",
      "year":2020,
      "contributor":"CSIRO x MSFT Synthetics",
      "date_created":"10/16/2020"
   }

licenses = [{
        "url": "",
        "id": 0,
        "name": "License"
    }]

categories = [
      {
         "supercategory":"H_beveragebottle",
         "id":1,
         "name":"H_beveragebottle"
      },
    {
         "supercategory":"D_lid",
         "id":2,
         "name":"D_lid"
      },
    {
         "supercategory":"S_cup",
         "id":3,
         "name":"S_cup"
      },
    
   ]

def createColorMaskImage2(instanceImage, instanceLabels, categoryKey='CategoryPath',
                          categoryToColorMapping=CATEGORIES, color_cat=color_cat):
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
        
        color = color_cat[color_counter]
        
        colorMaskImage[instanceImage==id,0] = color[0]
        colorMaskImage[instanceImage==id,1] = color[1]
        colorMaskImage[instanceImage==id,2] = color[2]
        
        category_ids[str(color)] = categoryValue
        color_counter += 1
    return colorMaskImage, category_ids

def load_image(path):
        return cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

def load_json(path):
    # Load json from file
    json_file = open(path)
    js = json.load(json_file)
    json_file.close()
    return js

def process_folder(params):
    global annotation_id, image_id, is_crowd, images, annotations, color_cat, CATEGORIES, cat_id

    #parses the results
    dataDirectory = params[0]
    outPathRoot = params[1]

    # parses the results
    results = sx.readDataDirectory(dataDirectory[2])

    #extracts the labeling metadata, rgb image, labeling image, depth
    instanceLabels = sx.parseInstancelabels(results['instance'][0])
    rgbImagePath = results['rgba'][1][0]
    instanceImagePath = results['instance'][1][0]

    # READ AND CREATE RGB Image
    rgbImage = sx.readRgbImage(rgbImagePath)
    rgb_path = outPathRoot+"/images/{:06d}.jpg".format(image_id) # e.g. 00000023.jpg
    sx.saveImage(rgbImage, rgb_path)
    

    # READ AND CREATE MASK (COLOR / BINARY)
    instanceImage = sx.readInstanceImage(instanceImagePath)
    outMaskImageForViewing, category_idS = createColorMaskImage2(instanceImage, instanceLabels, color_cat=color_cat)
    if category_idS == 'error':
        print("[removing file] >> mask processing error found in: ", rgbImagePath)
        return

    mask_path = outPathRoot+"/mask/{:06d}.png".format(image_id) # e.g. 00000023.jpg
    sx.saveImage(outMaskImageForViewing, mask_path)

    ### CREATE MASK ANNOTATIONS
    mask_image = Image.open(mask_path)
    sub_masks = create_sub_masks(mask_image)

    for color, sub_mask in sub_masks.items():
        category_id = cat_id[category_idS[color]]
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




def main(inPath, outPathRoot):
    global annotation_id, image_id, is_crowd, images, annotations, color_cat, CATEGORIES, cat_id
    global info, categories, licenses

    if not os.path.exists(outPathRoot):
        os.mkdir(outPathRoot)
        os.mkdir(outPathRoot+'/images')
        os.mkdir(outPathRoot+'/mask')

    dataDirectories = sx.listDataDirectories(inPath)

    for dataDirectory in dataDirectories:
        process_folder([dataDirectory, outPathRoot])

    print("processing done")

    print("saving annotations to coco as json ")
    ### create COCO JSON annotations
    my_dict = {}
    my_dict["info"]= info
    my_dict["licenses"]= licenses
    my_dict["images"]=images
    my_dict["categories"]=categories
    my_dict["annotations"]=annotations

    # TODO: specify coco file locaiton 
    output_file_path = os.path.join(outPathRoot,"coco_instances.json")
    with open(output_file_path, 'w+') as json_file:
        json_file.write(json.dumps(my_dict))

    print(">> complete. find coco json here: ", output_file_path)







# if run from command line it parsers the parameters and runs main function
if __name__ == "__main__":
    #if (len(sys.argv) != 3):
    #    print('Usage: run.py <IN> <OUT>')
    #    sys.exit(0)
    #inPathRoot = sys.argv[1]
    #outPathRoot = sys.argv[2]
    inPathRoot = '/home/redne/ZeroWaste3D/DataManager/sample_maya_data/raw'
    outPathRoot = '/home/redne/ZeroWaste3D/DataManager/sample_maya_data/output'
    #inPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds0/raw/9d15240302634bb99c11b4d275d410cd-bl1u8podu899269tckj3en4a24'
    #outPathRoot = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds0/images'
    main(inPathRoot, outPathRoot)
