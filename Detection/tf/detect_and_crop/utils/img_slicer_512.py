#https://github.com/plusbest/sat_img_slicer_512/blob/main/main.py
import os
import cv2
from image_slicer import slice      # pip install image-slicer
from PIL import Image

# import image
#im = cv2.imread('../images/original/000000001563.jpg')
#im = cv2.imread('../images/original/000000001563.jpg')
im = cv2.imread('original2/000000005576.jpg')

# get dims
h, w, c = im.shape

# Target width and height in pixels
#target_w = 512
#target_h = 512
target_w = 1024
target_h = 1024

# Incrementally trims dimension to fit target
# dimension ratio
def refactor(w, target_w):
    if w % target_w == 0:
        return w
    else:
        w = w - 1
        return refactor(w, target_w)


# Resizes image into target w and h
def crop(w, h):
    img_resize = cv2.resize(im, (w, h))
    print(img_resize.shape)
    # cv2.imshow("RESIZED", img_resize)
    # cv2.waitKey()
    return img_resize
# Splits image into specified pieces


def chopchop(fname, x_pieces, y_pieces):
    filename, file_extension = os.path.splitext(fname)
    im = Image.open(fname)
    imgwidth, imgheight = im.size
    height = imgheight // y_pieces
    width = imgwidth // x_pieces
    for i in range(0, y_pieces):
        for j in range(0, x_pieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            a = im.crop(box)
            try:
                a.save("./resized_test2_1024d/f" + filename + "-" + str(i) + "-" + str(j) + file_extension)
            except:
                pass


# Get workable resize dimensions
final_w = refactor(w, target_w)
final_h = refactor(h, target_h)

# Crop with new dimensions and save file
file_resized = crop(final_w, final_h)
cv2.imwrite("./resized.png", file_resized) 

# Calculate number of pieces (square)
x_pieces = final_w // target_w
y_pieces = final_h // target_h

# Call function to slice into pieces
fname = 'resized.png'
chopchop(fname, x_pieces, y_pieces)