# script that renames the current directory images 

import os
import cv2 as cv2

current_dir = os.getcwd()
write_dir = os.path.join(current_dir,'renamed')
count = 0
if(not os.path.exists(write_dir)):
    os.mkdir(write_dir)
for img_filename in os.listdir(current_dir):
    filename, extension = img_filename.split('.')
    img = cv2.imread(img_filename)
    out_filename = '.'.join([filename+'_'+str(count),extension])
    # print(filename+'_'+str(count)+extension)
    cv2.imwrite(os.path.join(write_dir,out_filename),img)
    count+=1