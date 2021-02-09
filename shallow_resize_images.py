## Bulk image resizer

# This script simply resizes all the images in a folder to one-eigth their
# original size. It's useful for shrinking large cell phone pictures down
# to a size that's more manageable for model training.

# Usage: place this script in a folder of images you want to shrink,
# and then run it.

import numpy as np
import cv2
import os

dir_path = os.getcwd()

for filename in os.listdir(dir_path):
    # If the images are not .JPG images, change the line below to match the image type.
    if filename.endswith(".JPG"):
        image = cv2.imread(filename)
        height,width,_ = image.shape
        print(height,width)
        if(height >= 1000 and width >= 1000):
            print(filename)
            resized = cv2.resize(image,None,fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            cv2.imwrite(filename,resized)
