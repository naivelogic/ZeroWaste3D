#%%
import sys
sys.path.append("../../") # go to parent dir

from utils.coco_manager import MaskManager

m = MaskManager(dataset_config_path = "/home/redne/mnt/project_zero/project_zero/ds1/experiments/dataset_config/")
m.set_resave_mask(new_mask_path="/home/redne/mnt/project_zero/project_zero/ds1/masks/")

train_set, val_set = m.make_datapath_list('/home/redne/mnt/project_zero/project_zero/ds1/parsed/', sample_amount=0.2)

test_val_paths = val_set
masklen = len(test_val_paths)-1
test_val_paths = test_val_paths[:masklen+1]
sample =  round(masklen * .05)

val_set, test_set = test_val_paths[-masklen+sample:], test_val_paths[:-masklen+sample]

print(f'train set count: {len(train_set)}\nvalidation set count: {len(val_set)}\ntest set count: {len(test_set)}')

###train set count: 399
###validation set count: 95
###test set count: 6

# Create Mask Disctionary for each of the dataset
#%%
def mask_runner(dataset_paths, mask_file_name):
    #dataset = val_set , train_set, test_set
    #ds_name = good name to name the mask_dic
    m = MaskManager(dataset_config_path = "/home/redne/mnt/project_zero/project_zero/ds1/experiments/dataset_config/")
    m.set_resave_mask(new_mask_path="/home/redne/mnt/project_zero/project_zero/ds1/masks/")

    m.custom_classes_flag = True
    m.custom_classes = {"fork":"utensils", "spoon":"utensils", "knife":"utensils", 'coffeeCup':'coffeeCup', 'clearCup':'clearCup'}
    m.colorMapping = {"utensils":[0,255,0], 'coffeeCup':[255,0,0], 'clearCup':[0,0,255]}
    m.mask_colors = { "utensils":(0,255,0), 'coffeeCup':(255,0,0), 'clearCup':(0,0,255)}
    m.super_categories = {"utensils":"utensils", 'coffeeCup':'coffeeCup', 'clearCup':'clearCup'}
    m.get_super_categories = {"utensils":["utensils"], 'coffeeCup': ['coffeeCup'],'clearCup':['clearCup']}
    
    m.start(phase=mask_file_name, mask_paths=dataset_paths)

    print(f'there are {len(m.masks)} images')
    m.show_mask_img(len(m.masks)-1)
    m.write_masks_to_json(phase=mask_file_name)

#%%
mask_runner(train_set, 'ds1_3class_train')

#%%
mask_runner(val_set, 'ds1_3class_val')
mask_runner(test_set, 'ds1_3class_test')

# Create COCO Dataset Files
#%%
from utils import coco_json_utils as cjutils
cjc = cjutils.CocoJsonCreator()

mask_file_name='ds1_3class_test'
class my_args:
    mask_definition = f'/home/redne/mnt/project_zero/project_zero/ds1/experiments/dataset_config/{mask_file_name}_mask_definitions.json'
    dataset_info = '/home/redne/mnt/project_zero/project_zero/ds1/experiments/dataset_config/dataset_info.json'
    phase = mask_file_name
    dataset_config_path = f'/home/redne/mnt/project_zero/project_zero/ds1/experiments/dataset_config/'
args = my_args()

cjc.main(args)

