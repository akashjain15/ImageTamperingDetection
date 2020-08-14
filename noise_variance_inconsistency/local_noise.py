import sys
import math
import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage.restoration import estimate_sigma

img = cv2.imread('..//images//tes.jpg',0)

blocks = []
variances = []
imgwidth, imgheight = img.shape[0:2]
blockSize = 64
print (img.shape)
#break up image into NxN blocks, N = blockSize
for j in range(0,imgheight,blockSize):
    for i in range(0,imgwidth,blockSize):
        a = img
        if (i + blockSize > imgwidth and j + blockSize > imgheight):
            a = img[i:,j:]
        elif (i + blockSize > imgwidth):
            a = img[i:,j:j+blockSize]
        elif (j + blockSize > imgheight):
            a = img[i:i+blockSize,j:]
        else:
            a = img[i:i+blockSize,j:j+blockSize]
        
        sigma = estimate_sigma(a, multichannel=False, average_sigmas=True)
        blocks.append(a)
        variances.append([sigma])

print(variances)
kmeans = KMeans(n_clusters=2, random_state=0).fit(variances)
center1, center2 = kmeans.cluster_centers_
sigma = estimate_sigma(img, multichannel=False, average_sigmas=True)
print(sigma)
print(center1,center2)
