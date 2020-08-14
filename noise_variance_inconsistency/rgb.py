import sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.restoration import estimate_sigma

def initClusters(data):
    clusters = []

    noOfPts = len(data)
    print('---')
    print('no of pts', noOfPts)
    x = int(noOfPts / 2)
    print(x)

    for i in range(2):
        s = 0
        offset = i * x

        s = np.sum(data[offset: offset+x])
        print(x)
        print(s)
        clusters.append(s/x)

    return clusters

def updateClusters(oldClusters, data):
    totalPtsInCluster = []
    clusters = []

    for i in range(0, 2):
        totalPtsInCluster.append(0)
        clusters.append(0.0)


    for pt in data:
        dist = []

        for cluster in oldClusters:
            dist.append(np.abs(cluster - pt))
            
        idx = dist.index(min(dist))
        totalPtsInCluster[idx] += 1
        clusters[idx] += pt

    print('------')
    print(totalPtsInCluster)
    for i in range(0, 2):

        if (totalPtsInCluster[0] == 0):
            clusters[0] = clusters[1]
            totalPtsInCluster[0] = totalPtsInCluster[1]
        if (totalPtsInCluster[1] == 0):
            clusters[1] = clusters[0]
            totalPtsInCluster[1] = totalPtsInCluster[0]

        clusters[i] = float(clusters[i] / totalPtsInCluster[i])

    return clusters

def clusterData(data):
    print('---------')
    clusters = initClusters(data)
    print(clusters)

    oldClusters = []

    for i in range(10):
        oldClusters = clusters
        clusters = updateClusters(oldClusters, data)

    print ('------------')
    print (clusters)
    return clusters

img = cv2.imread('..//4cam_splc//canong3_canonxt_sub_03.tif')
#img = cv2.imread('..//4cam_auth//canong3_02_sub_08.tif')

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


print('BLUE')
kmeans_b = KMeans(n_clusters=2).fit(variances_b)
center1_b, center2_b = kmeans_b.cluster_centers_
if(center2_b < center1_b):
    center1_b, center2_b = center2_b, center1_b
sigma_b = estimate_sigma(imgb, multichannel=False, average_sigmas=True)
print(sigma_b)
print(center1_b,center2_b)
pt_cen1_b = 0
pt_cen2_b = 0
for i in variances_b:
    cen1 = abs(center1_b-i)
    cen2 = abs(center2_b-i)
    if(cen1<cen2):
        pt_cen1_b += 1
    else :
        pt_cen2_b +=1
print(pt_cen1_b,pt_cen2_b)

print('GREEN')
kmeans_g = KMeans(n_clusters=2).fit(variances_g)
center1_g, center2_g = kmeans_g.cluster_centers_
if(center2_g < center1_g):
    center1_g, center2_g = center2_g, center1_g
sigma_g = estimate_sigma(imgg, multichannel=False, average_sigmas=True)
print(sigma_g)
print(center1_g,center2_g)
pt_cen1_g = 0
pt_cen2_g = 0
for i in variances_g:
    cen1 = abs(center1_g-i)
    cen2 = abs(center2_g-i)
    if(cen1<cen2):
        pt_cen1_g += 1
    else :
        pt_cen2_g +=1
print(pt_cen1_g,pt_cen2_g)

print('RED')
kmeans_r = KMeans(n_clusters=2).fit(variances_r)
center1_r, center2_r = kmeans_r.cluster_centers_
# center1_r, center2_r = clusterData(variances_r)

if(center2_r < center1_r):
    center1_r, center2_r = center2_r, center1_r
sigma_r = estimate_sigma(imgr, multichannel=False, average_sigmas=True)
print(sigma_r)
print(center1_r,center2_r)
pt_cen1_r = 0
pt_cen2_r = 0
for i in variances_r:
    cen1 = abs(center1_r-i)
    cen2 = abs(center2_r-i)
    if(cen1<cen2):
        pt_cen1_r += 1
    else :
        pt_cen2_r +=1
print(pt_cen1_r,pt_cen2_r)

#plt.title('Spliced Image')
plt.plot(number,variances_b,'b.')
plt.plot(number,variances_g,'g.')
plt.plot(number,variances_r,'r.')
#plt.ylabel('Value of sigmas')
#plt.ylim(0,40)
plt.show()