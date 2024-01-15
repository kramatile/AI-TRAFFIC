import os 
import cv2
import numpy as np
# declare the lists to return : 
IMG_WIDTH = 30
IMG_HEIGHT = 30
images = []
label = []
    # loop over the directories : 
for i in range(3) :
    images_dir = os.listdir(os.path.join('.','gtsrb-small',str(i)))
    for image in images_dir :
         # read the img using cv2
        img = cv2.imread(os.path.join('.','gtsrb-small',str(i),image))
            # declare the shape variables
        height = img.shape[0]
        width  = img.shape[1]
            # revise the shape of the image
        if height != IMG_HEIGHT or width != IMG_WIDTH :
            img = cv2.resize(img, (IMG_HEIGHT,IMG_WIDTH), interpolation = cv2.INTER_AREA)
            # stock it as a multi-dimetionnal array
        images.append(img)
        label.append(i)
    # return the final lists (hopefully)
    the_same = True
    for image in images : 
        if image.shape[0] != IMG_HEIGHT or image.shape[1] != IMG_WIDTH :
            the_same = False
    print(the_same)