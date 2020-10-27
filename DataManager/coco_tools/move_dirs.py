import shutil, os
import numpy as np

PATH_TO_TEST_IMAGES_DIR = '/mnt/omreast_users/phhale/csiro_trashnet/original_samples/Subset_Trashnet'
TEST_IMAGE_PATHS= os.listdir(PATH_TO_TEST_IMAGES_DIR)

TEST_IMAGE_PATHS = np.random.choice(TEST_IMAGE_PATHS, 15)

SAVE_PATH = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/csior_test_images/ds0v5_detector_images/csiro_crop_sample_02/original'
os.makedirs(SAVE_PATH, exist_ok=True)

for image_path in TEST_IMAGE_PATHS:
    source = os.path.join(PATH_TO_TEST_IMAGES_DIR, image_path)
    target = os.path.join(SAVE_PATH, image_path)
    shutil.copy(source, target)