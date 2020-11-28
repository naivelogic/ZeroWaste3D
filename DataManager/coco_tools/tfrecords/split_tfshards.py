import shutil, os
import numpy as np
from tqdm import tqdm

#NEW_TF_RECORD_PATH = 'tfrecords_s30_r70' #'tfrecords_s70_r30' 
#TF_PREFIX = 'csiro_crop~train'
#TF_PREFIX_NUM_SHARD = 1040

#TF_PREFIX ='csiro_crop~test'
#TF_PREFIX_NUM_SHARD = 234

#TF_PREFIX = 'ds2_storm~train' # done s30
#TF_PREFIX_NUM_SHARD = 3196

#TF_PREFIX = 'ds2_storm~test'
#TF_PREFIX_NUM_SHARD = 719

#num = 50

for image_id in tqdm(range(0,num)):
    SOURCE = f'/mnt/omreast_users/phhale/csiro_trashnet/datasets/crop_ds0/coco_ds_csiro/tfrecords/{TF_PREFIX}-{image_id:05d}-of-{TF_PREFIX_NUM_SHARD:05d}.tfrecord'
    TARGET = f'/mnt/omreast_users/phhale/csiro_trashnet/datasets/crop_ds0/coco_ds_csiro/{NEW_TF_RECORD_PATH}/{TF_PREFIX}-{image_id:05d}-of-{num:05d}.tfrecord'
    shutil.copy(SOURCE, TARGET)
    
print("complete with: ", TARGET)
    