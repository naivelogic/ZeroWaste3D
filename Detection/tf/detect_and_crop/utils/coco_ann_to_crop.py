#https://github.com/AsharFatmi/Utility_functions_python/blob/master/CropImagesUsingCOCOAnnotation.py

import os
import json
from PIL import Image


def main():

    annotation_path = '/home/redne/ZeroWaste3D/DataManager/dev/sample_maya_data/output_ds2/coco_instances.json'
    img_path = '/home/redne/ZeroWaste3D/DataManager/dev/sample_maya_data/output_ds2/images'
    save_dir = '/home/redne/ZeroWaste3D/DataManager/dev/sample_maya_data/output_ds2/cropped'

    with open(annotation_path, 'r') as myfile:
        X=myfile.read()
    obj = json.loads(X)

    i = 1

    for annotation in obj['annotations']:
        #print(annotation)
        image_id = annotation['image_id']
        #print(image_id)

        for id in obj['images']:
            if (id['id'] == image_id):
                #print('True '+ str(image_id))
                imgName = id['file_name']
                #imgName = img.split('/')
                #print(os.path.join(img_path,imgName[1]))
                imageObject  = Image.open(os.path.join(img_path,imgName))
                
                x = annotation['bbox'][0]
                y = annotation['bbox'][1]
                width = annotation['bbox'][2]
                height = annotation['bbox'][3]
                
                cropped = imageObject.crop((x, y, x+width, y+height))
                cropped.save(os.path.join(save_dir, '{}.jpg'.format(i)))
                i += 1


if __name__ == "__main__":
    main()