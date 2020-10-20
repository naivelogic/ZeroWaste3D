import random, shutil, os
copy_folder_imgs = '/mnt/omreast_users/phhale/csiro_trashnet/original_samples/Subset_Trashnet/'
copy_images = os.listdir(copy_folder_imgs)

MOVE_IMG_PATH = 'original2/'
samples_test_images = random.choices(copy_images, k=10)
for f in samples_test_images:
    current = os.path.join(copy_folder_imgs,f)
    target = os.path.join(MOVE_IMG_PATH,f)
    shutil.copy(current, target)
    
print("done.")