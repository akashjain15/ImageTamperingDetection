# Generating Single Compressed images from Uncompressed Images

import numpy as np
import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

images = load_images_from_folder('../ucid/f1')
k = 0
for image in images:
     firstq = 20
     k += 1
     for x in range(0,8):
          firstq += 10
          cv2.imwrite("../single_comp/img"+str(k)+"comp"+str(firstq)+".jpg",image,[int(cv2.IMWRITE_JPEG_QUALITY),firstq])


