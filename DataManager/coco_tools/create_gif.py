import os, sys
from PIL import Image
from tqdm import tqdm
# Open all the frames
gif_images = []
#image_folder = '/home/redne/ZeroWaste3D/Detection/tf/detect_and_crop/utils/detectv1_to_gif/'
#image_folder = '/mnt/omreast_users/phhale/zerowaste/03-experiments/ds2/yolact/yolact_d/yolact_r50_ds2_x0/output_images/redwest_test_images_01/'
#image_folder = '/home/redne/ZeroWaste3D/DataManager/dev/sample_frames_ds05/sample_frames_ds05/'
#image_folder = '/mnt/csiro/ds0_categorydetector/positives/show_predictions/'
#image_folder = '/home/redne/ZeroWaste3D/Detection/tf/detect_and_crop/utils/dev/output/ds0v5_detector/'

#image_folder = '/mnt/omreast_users/phhale/csiro_trashnet/original_samples/Validation_v0/synthetic0_real100/' #'/mnt/csiro/ds0v5_categorydetector/positives/show_predictions/'
#gif_frames = os.listdir(image_folder)
#import glob
#gif_frames=  glob.glob(os.path.join(image_folder, '**', '*.JPG*'), recursive=True)  # for Validation csiro samples

#image_folder = '/home/redne/ZeroWaste3D/Detection/tf/ds0v5_detector_images/csiro_crop_sample_02/original_bbox/'
image_folder = '/home/redne/WaterWaste/test_images/high_res_val_set_013/bbox'
gif_frames = os.listdir(image_folder)

# sub sample the images in the folder for the gif
import numpy as np
#gif_frames = np.random.choice(gif_frames, 30)
#gif_frames = np.random.choice(gif_frames, 130)
for n in tqdm(gif_frames):
    frame = Image.open(os.path.join(image_folder,n))
    gif_images.append(frame)

# Save the frames as an animated GIF
#gif_images[0].save('../output_results/output_images_test_2/output_images_test_2.gif',
#print(os.listdir())
#gif_images[0].save('outputs/output_images_test_2.gif',
#gif_images[0].save('outputs/ds0v5_detection_bbox_csiro_sample02.gif',
DURATION = 1500 # Default
#LOOP = 50 # Default
LOOP = 5
#gif_images[0].save('/mnt/omreast_users/phhale/csiro_trashnet/original_samples/Validation_v0/csiro_output_gif/synthetic0_real100_ds0v5_110620.gif',
gif_images[0].save('test_val1.gif',
               save_all=True,
               append_images=gif_images[1:],
               duration=DURATION,
               loop=LOOP)