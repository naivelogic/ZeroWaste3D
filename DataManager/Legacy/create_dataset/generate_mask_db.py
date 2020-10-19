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

# debug for nocs - 7/15
DATASET_PATH = f'/mnt/daredevildiag/6PACK/z3d/ds1/InstanceGroup2Desccamera_0camera_Shape0_iter0/'
DATASET_RAW_FOLDER = f'/mnt/daredevildiag/6PACK/z3d/ds1/'

# debug for water waste - 10/15/20
DATASET_PATH = f'/home/redne/ZeroWaste3D/DataManager/create_dataset/sample_maya_raw/ds1/'
DATASET_RAW_FOLDER = f'/home/redne/ZeroWaste3D/DataManager/create_dataset/sample_maya_raw/ds1/raw/'



from sklearn.model_selection import train_test_split
dir_list = os.listdir(DATASET_RAW_FOLDER)
train, val = train_test_split(dir_list, test_size=0.2, random_state=31)
val, test = train_test_split(val, test_size=0.1, random_state=31)

print(f'training dataset size: {len(train)}\nvalidation dataset size: {len(val)}\ntest dataset size: {len(test)}')



#%%
# debug for nocs - 7/15
#train = ['/mnt/daredevildiag/6PACK/z3d/ds1/raw/InstanceGroup2Desccamera_0camera_Shape0_iter0/']
train = ['InstanceGroup2Desccamera_0camera_Shape0_iter0']
#DATASET_PATH = '/mnt/daredevildiag/6PACK/z3d/ds1/'
DATASET_PATH = f'/home/redne/ZeroWaste3D/DataManager/create_dataset/sample_maya_raw/ds1/'
def mask_runner(dataset_paths, phase):
    m = MaskManager(DATASET_PATH)

    m.dataset_raw_folder = os.path.join(DATASET_PATH, 'raw')

    # save new mask & images
    m.custom_classes_flag = True
    m.resave_masks = True
    m.resave_mask_path = os.path.join(DATASET_PATH, 'masks')

    m.resave_images_flag = True
    m.resave_images_path = os.path.join(DATASET_PATH, 'images')


    m.custom_classes = {
            "H_beveragebottle": "H_beveragebottle",
            "D_lid": "D_lid",
            "S_cup": "S_cup"
            }
    m.colorMapping = {
            "H_beveragebottle": [0, 255, 0],
            'D_lid': [255, 0, 0],
            'S_cup': [0, 0, 255]
        }
    m.mask_colors = {
            "H_beveragebottle": (0, 255, 0),
            "D_lid": (255, 0, 0),
            "S_cup": (0, 0, 255),
        }
    m.super_categories = {
            "H_beveragebottle": "H_beveragebottle",
            "D_lid": "D_lid",
            "S_cup": "S_cup"
            }
    m.get_super_categories = {
            "H_beveragebottle": ["H_beveragebottle"],
            "D_lid": ["D_lid"],
            "S_cup": ["S_cup"]
        }
    
    m.start(phase=phase, mask_paths=dataset_paths)

    print(f'there are {len(m.masks)} images')
    m.show_mask_img(len(m.masks)-1)
    m.write_masks_to_json(phase=phase)

#%%
mask_runner(train, 'train')    # debug for nocs - 7/15
#mask_runner(train, 'ds2_3c_train')

#%%
mask_runner(test, 'ds2_3c_test')

#%%
mask_runner(val, 'ds2_3c_val')


"""
python coco_json_utils.py -md /mnt/zerowastepublic/02-datasets/ds2/dataset_config/ds2_3c_test_mask_definitions.json -di /home/redne/mnt/project_zero/project_zero/ds1/experiments/dataset_config/dataset_info.json -ph ds2_3c_test -dp /mnt/zerowastepublic/02-datasets/ds2/dataset_config/


python coco_json_utils.py -md /mnt/daredevildiag/6PACK/z3d/ds1/dataset_config/train_mask_definitions.json -di dataset_info.json -ph train -dp /mnt/daredevildiag/6PACK/z3d/ds1/dataset_config/
"""