import sys
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.restoration import estimate_sigma


def load_filenames_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = str(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def detect(path):
    #st = str(path)
    img = cv2.imread(path)
    imgwidth, imgheight = img.shape[0:2]
    blockSize = 16

    imgb = img[:,:,0]
    imgg = img[:,:,1]
    imgr = img[:,:,2]

    blocks_b = []
    blocks_g = []
    blocks_r = []

    variances_b = []
    variances_g = []
    variances_r = []

    num = 0
    number = []

    for j in range(0,imgheight,blockSize):
        for i in range(0,imgwidth,blockSize):
            b = imgb
            g = imgg
            r = imgr
            if (i + blockSize > imgwidth and j + blockSize > imgheight):
                b = imgb[i:,j:]
                g = imgg[i:,j:]
                r = imgr[i:,j:]
            elif (i + blockSize > imgwidth):
                b = imgb[i:,j:j+blockSize]
                g = imgg[i:,j:j+blockSize]
                r = imgr[i:,j:j+blockSize]
            elif (j + blockSize > imgheight):
                b = imgb[i:i+blockSize,j:]
                g = imgg[i:i+blockSize,j:]
                r = imgr[i:i+blockSize,j:]
            else:
                b = imgb[i:i+blockSize,j:j+blockSize]
                g = imgg[i:i+blockSize,j:j+blockSize]
                r = imgr[i:i+blockSize,j:j+blockSize]
            
            blocks_b.append(b)
            blocks_g.append(g)
            blocks_r.append(r)

            num += 1
            number.append(num)

            sigma_b = estimate_sigma(b, multichannel=False, average_sigmas=True)
            if (np.isnan(sigma_b)):
                variances_b.append([0.0])
            else:
                variances_b.append([sigma_b])

            sigma_g = estimate_sigma(g, multichannel=False, average_sigmas=True)
            if (np.isnan(sigma_g)):
                variances_g.append([0.0])
            else:
                variances_g.append([sigma_g])

            sigma_r = estimate_sigma(r, multichannel=False, average_sigmas=True)
            if (np.isnan(sigma_r)):
                variances_r.append([0.0])
            else:
                variances_r.append([sigma_r])

    #print('BLUE')
    kmeans_b = KMeans(n_clusters=2).fit(variances_b)
    center1_b, center2_b = kmeans_b.cluster_centers_
    if(center2_b < center1_b):
        center1_b, center2_b = center2_b, center1_b
    sigma_b = estimate_sigma(imgb, multichannel=False, average_sigmas=True)
    #st = st+','+str(sigma_b)+','+str(float(center1_b))+','+str(float(center2_b))
    pt_cen1_b = 0
    pt_cen2_b = 0
    for i in variances_b:
        cen1 = abs(center1_b-i)
        cen2 = abs(center2_b-i)
        if(cen1<cen2):
            pt_cen1_b += 1
        else :
            pt_cen2_b +=1
    #st = st+','+str(pt_cen1_b)+','+str(pt_cen2_b)

    #print('GREEN')
    kmeans_g = KMeans(n_clusters=2).fit(variances_g)
    center1_g, center2_g = kmeans_g.cluster_centers_
    if(center2_g < center1_g):
        center1_g, center2_g = center2_g, center1_g
    sigma_g = estimate_sigma(imgg, multichannel=False, average_sigmas=True)
    #st = st+','+str(sigma_g)+','+str(float(center1_g))+','+str(float(center2_g))
    pt_cen1_g = 0
    pt_cen2_g = 0
    for i in variances_g:
        cen1 = abs(center1_g-i)
        cen2 = abs(center2_g-i)
        if(cen1<cen2):
            pt_cen1_g += 1
        else :
            pt_cen2_g +=1
    #st = st+','+str(pt_cen1_g)+','+str(pt_cen2_g)

    #print('RED')
    kmeans_r = KMeans(n_clusters=2).fit(variances_r)
    center1_r, center2_r = kmeans_r.cluster_centers_
    if(center2_r < center1_r):
        center1_r, center2_r = center2_r, center1_r
    sigma_r = estimate_sigma(imgr, multichannel=False, average_sigmas=True)
    #st = st+','+str(sigma_r)+','+str(float(center1_r))+','+str(float(center2_r))
    pt_cen1_r = 0
    pt_cen2_r = 0
    for i in variances_r:
        cen1 = abs(center1_r-i)
        cen2 = abs(center2_r-i)
        if(cen1<cen2):
            pt_cen1_r += 1
        else :
            pt_cen2_r +=1
    #st = st+','+str(pt_cen1_r)+','+str(pt_cen2_r)+'\n'
    #fl.write(st)
    avgn = (sigma_b + sigma_g + sigma_r)/3
    avgc1 = (center1_b + center1_g + center1_r)/3
    avgc2 = (center2_b + center2_g + center2_r)/3
    p1 = int((pt_cen1_b + pt_cen1_g + pt_cen1_r)/3) 
    p2 = int((pt_cen2_b + pt_cen2_g + pt_cen2_r)/3)

    if(p2 < ((pt_cen1_b+pt_cen2_b)*0.25)):
        return True
    else:
        return False

images = load_filenames_from_folder('..//4cam_splc')
#images = load_filenames_from_folder('..//4cam_auth')
# Columbia uncompressed image splicing detection evaluation dataset 2006

Total = 0
Tampered = 0
Untampered = 0
#fl = open('4cam_splc.csv','w')
#fl = open('4cam_auth.csv','w')

for image in images:
    #check = detect(image,fl)
    check = detect(image)
    Total += 1
    if(check):
        Tampered += 1
    else:
        Untampered += 1

#fl.close()
print('Total Images:\t',Total)
print('Tampered Images:\t',Tampered)
print ('UnTampered Images:\t',Untampered)

acc = (Tampered/Total)*100
print('Acc:\t',acc)

