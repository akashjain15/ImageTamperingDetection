# Generating Double Compressed Images from Single Compressed Images

import numpy as np
import cv2
import os


k = 88
for x in range(0,8):
    k+=1
    for firstq in range(30,90,10):
        image = cv2.imread("../single_comp/f12/img"+str(k)+"comp"+str(firstq)+".jpg")
        secondq = 30
        for i in range(0,7):
            if(firstq!=secondq):
                cv2.imwrite("../doub_comp/img"+str(k)+"comp"+str(firstq)+str(secondq)+".jpg",image,[int(cv2.IMWRITE_JPEG_QUALITY),secondq])
            secondq += 10

            

