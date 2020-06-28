#%%
import sys, os, glob
import numpy as np
sys.path.append("../") # go to parent dir

from utils.coco_manager import MaskManager


## Create Train/Val/Test/ Datasets
#Train:  80% 
#Val:    18%
#Test:    2%

#%%
DATASET_VERSON = 'ds2'
DATASET_PATH = f'/mnt/zerowastepublic/02-datasets/{DATASET_VERSON}/'
DATASET_RAW_FOLDER = f'/mnt/zerowastepublic/02-datasets/{DATASET_VERSON}/raw/'

from sklearn.model_selection import train_test_split
dir_list = os.listdir(DATASET_RAW_FOLDER)
train, val = train_test_split(dir_list, test_size=0.2, random_state=31)
val, test = train_test_split(val, test_size=0.1, random_state=31)

print(f'training dataset size: {len(train)}\nvalidation dataset size: {len(val)}\ntest dataset size: {len(test)}')



#%%

def mask_runner(dataset_paths, phase):
    m = MaskManager(DATASET_PATH)

    m.dataset_raw_folder = os.path.join(DATASET_PATH, 'raw')

    # save new mask & images
    m.custom_classes_flag = True
    m.resave_masks = True
    m.resave_mask_path = os.path.join(DATASET_PATH, 'masks')

    m.resave_images_flag = True
    m.resave_images_path = os.path.join(DATASET_PATH, 'images')


    m.custom_classes = {"fork":"utensils", "spoon":"utensils", "knife":"utensils", 'coffeeCup':'coffeeCup', 'clearCup':'clearCup'}
    m.colorMapping = {"utensils":[0,255,0], 'coffeeCup':[255,0,0], 'clearCup':[0,0,255]}
    m.mask_colors = { "utensils":(0,255,0), 'coffeeCup':(255,0,0), 'clearCup':(0,0,255)}
    m.super_categories = {"utensils":"utensils", 'coffeeCup':'coffeeCup', 'clearCup':'clearCup'}
    m.get_super_categories = {"utensils":["utensils"], 'coffeeCup': ['coffeeCup'],'clearCup':['clearCup']}
    
    m.start(phase=phase, mask_paths=dataset_paths)

    print(f'there are {len(m.masks)} images')
    m.show_mask_img(len(m.masks)-1)
    m.write_masks_to_json(phase=phase)

#%%
mask_runner(train, 'ds2_3c_train')

#%%
mask_runner(test, 'ds2_3c_test')

#%%
mask_runner(val, 'ds2_3c_val')


"""
python coco_json_utils.py -md /mnt/zerowastepublic/02-datasets/ds2/dataset_config/ds2_3c_test_mask_definitions.json -di /home/redne/mnt/project_zero/project_zero/ds1/experiments/dataset_config/dataset_info.json -ph ds2_3c_test -dp /mnt/zerowastepublic/02-datasets/ds2/dataset_config/
"""