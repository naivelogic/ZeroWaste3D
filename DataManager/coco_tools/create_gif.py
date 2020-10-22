import os, sys
from PIL import Image

# Open all the frames
gif_images = []
#image_folder = '/home/redne/ZeroWaste3D/Detection/tf/detect_and_crop/utils/detectv1_to_gif/'
image_folder = '/mnt/omreast_users/phhale/zerowaste/03-experiments/ds2/yolact/yolact_d/yolact_r50_ds2_x0/output_images/redwest_test_images_01/'
gif_frames = os.listdir(image_folder)
for n in gif_frames:
    frame = Image.open(os.path.join(image_folder,n))
    gif_images.append(frame)

# Save the frames as an animated GIF
#gif_images[0].save('../output_results/output_images_test_2/output_images_test_2.gif',
print(os.listdir())
gif_images[0].save('outputs/output_images_test_2.gif',
               save_all=True,
               append_images=gif_images[1:],
               duration=1500,
               loop=50)
