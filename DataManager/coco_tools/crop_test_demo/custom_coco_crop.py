# https://github.com/microsoft/CameraTraps/blob/master/research/active_learning/data_preprocessing/crop_images_from_coco_bboxes.py


'''
Produces a directory of crops from a COCO-annotated .json full of 
bboxes.
'''
import numpy as np
import argparse, ast, csv, json, pickle, os, sys, time, tqdm, uuid
from PIL import Image
#sys.path.append("../")
#
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"../")
#sys.path.append("/home/redne/ZeroWaste3D/DataManager/")
#from dataset_configs.waterwaste_ds_config import *
from tqdm import tqdm

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def main():
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True, type=str, help='Path to a directory containing full-sized images in the dataset.')
    parser.add_argument('--coco_json', required=True, type=str, help='Path to COCO JSON file for the dataset.')
    parser.add_argument('--crop_dir', required=True, type=str, help='Path to output directory for crops.')
    parser.add_argument('--padding_factor', type=float, default=1.3, help='We will crop a tight square box around the animal enlarged by this factor. ' + \
                   'Default is 1.3 * 1.3 = 1.69, which accounts for the cropping at test time and for' + \
                    ' a reasonable amount of context')
    args = parser.parse_args()
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
    
    IMAGE_DIR = args.image_dir
    COCO_JSON = args.coco_json
    CROP_DIR = args.crop_dir
    PADDING_FACTOR = args.padding_factor
    """

    IMAGE_DIR = 'original_images/'
    COCO_JSON = 'coco_instances.json'
    CROP_DIR = 'crop_images'   
    CROP_PEFIX = 'ds2_storm_crop_'
    
    #PADDING_FACTOR = 2
    PADDING_FACTOR = 1
    

    if not os.path.exists(CROP_DIR):
        os.mkdir(CROP_DIR)
    
    #if not os.path.exists(CROP_DIR+"images"):
    #    os.mkdir(CROP_DIR+"images")


    #PADDING_FACTOR = 4
    crops_json = {}
    my_dict = {}

    # These ids will be automatically increased as we go
    coco_annotation_id = 1
    coco_image_id = 1
    coco_is_crowd = 0
    coco_annotations = list()
    coco_images = list()

    coco_json_data = json.load(open(COCO_JSON, 'r'))
    info = coco_json_data['info']
    licenses = coco_json_data['licenses']
    categories = coco_json_data['categories']
    images = {im['id']:im for im in coco_json_data['images']}
    bboxes_available = any([('bbox' in a.keys()) for a in coco_json_data['annotations']])
    assert bboxes_available, 'COCO JSON does not contain bounding boxes, need to run a detector first.'
    
    crop_counter = 0
    timer = time.time()
    for ann in tqdm(coco_json_data['annotations']):
        if 'bbox' not in ann.keys():
            continue
        image_id = ann['image_id']
        image_fn = images[image_id]['file_name'].replace('\\', '/')
        img = np.array(Image.open(os.path.join(IMAGE_DIR, image_fn)))
        if img.dtype != np.uint8:
            print('Failed to load image '+ image_fn)
            continue
        crop_counter += 1

        image_height = images[image_id]['height']
        image_width = images[image_id]['width']
        image_grayscale = bool(np.all(abs(np.mean(img[:,:,0]) - np.mean(img[:,:,1])) < 1) & (abs(np.mean(img[:,:,1]) - np.mean(img[:,:,2])) < 1))


        detection_box_pix = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]
        detection_box_size = np.vstack([detection_box_pix[2] - detection_box_pix[0], detection_box_pix[3] - detection_box_pix[1]]).T
        offsets = (PADDING_FACTOR*np.max(detection_box_size, keepdims=True) - detection_box_size)/2
        crop_box_pix = detection_box_pix + np.hstack([-offsets,offsets])
        crop_box_pix = np.maximum(0,crop_box_pix).astype(int)
        crop_box_pix = crop_box_pix[0]
        detection_padded_cropped_img = img[crop_box_pix[1]:crop_box_pix[3], crop_box_pix[0]:crop_box_pix[2]]
        
        crop_fn = os.path.join(CROP_DIR,CROP_PEFIX+"{:06d}.jpg".format(coco_image_id)) 
        crop_width = int(detection_padded_cropped_img.shape[1])
        crop_height = int(detection_padded_cropped_img.shape[0])
        crop_rel_size = (crop_width*crop_height)/(image_width*image_height)
        #detection_conf = 1 # for annotated crops, assign confidence of 1

        Image.fromarray(detection_padded_cropped_img).save(crop_fn)

        x_offset, y_offset = [round(x) for x in list(offsets[0])]
        bbox_x, bbox_y, bbox_w, bbox_h = [x_offset, y_offset, (crop_width - x_offset*2),  (crop_height - y_offset*2)]
         
        annotation = {
            'iscrowd': coco_is_crowd,
            'image_id': coco_image_id,
            'category_id': ann['category_id'],
            'id': coco_annotation_id,
            'bbox':[bbox_x, bbox_y, bbox_w, bbox_h],
            'area': 0,
            'relative_size': crop_rel_size,
            'offsets': list(offsets[0])
        }
        coco_annotations.append(annotation)
        coco_annotation_id += 1
  
        new_img={
            'license': 0,
            'file_name': crop_fn,
            'width':crop_width,
            'height': crop_height,
            'id': coco_image_id,
            'source_file_name': image_fn
        }
        coco_images.append(new_img)

        coco_image_id += 1

    print("saving annotations to coco as json ")
    ### create COCO JSON annotations
    
    my_dict["info"]= info
    my_dict["licenses"]= licenses
    my_dict["images"]=coco_images
    my_dict["categories"]=categories
    my_dict["annotations"]=coco_annotations

    with open(os.path.join(CROP_DIR, CROP_PEFIX+'coco_instances.json'), 'w') as outfile:
            json.dump(my_dict, outfile)

            
if __name__ == '__main__':
    main()
    