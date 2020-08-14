import sys
import numpy as np
from scipy import fftpack as fftp
import cv2
from matplotlib import pyplot as plt
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def detect(folder):
    thres = 0.5
    images = load_images_from_folder(folder)
    Total_Images_count = 0
    Double_compressed = 0
    Single_compressed = 0
    Uncompressed = 0
    for image in images:
        dct_rows = 0
        dct_cols = 0
        shape = image.shape

        if shape[0]%8 != 0:dct_rows = shape[0]+8-shape[0]%8
        else:dct_rows = shape[0]
        if shape[1]%8 != 0:dct_cols = shape[1]+8-shape[1]%8
        else:dct_cols = shape[1]

        dct_image = np.zeros((dct_rows,dct_cols,3),np.uint8)
        dct_image[0:shape[0], 0:shape[1]] = image

        y = cv2.cvtColor(dct_image,cv2.COLOR_BGR2YCR_CB)[:,:,0]
        h=y.shape[0]
        Y = y.reshape(h//8,8,-1,8).swapaxes(1,2).reshape(-1, 8, 8)

        qDCT =[]
        for i in range(0,Y.shape[0]): 
            qDCT.append(cv2.dct(np.float32(Y[i])))

        qDCT = np.asarray(qDCT, dtype=np.float32)
        qDCT = np.rint(qDCT - np.mean(qDCT, axis = 0)).astype(np.int32)

        f,a1 = plt.subplots(8,8)
        a1 = a1.ravel()
        k=0
        flag = 0
        pcount = []
        for idx,ax in enumerate(a1):
            k+=1
            data = qDCT[:,int(idx/8),int(idx%8)]
            val,key = np.histogram(data, bins=np.arange(data.min(), data.max()+1),normed = True)
            z = np.absolute(fftp.fft(val))
            z = np.reshape(z,(len(z),1))
            rotz = np.roll(z,int(len(z)/2))
            slope = rotz[1:] - rotz[:-1]
            indices = [i+1 for i in range(len(slope)-1) if slope[i] > 0 and slope[i+1] < 0]
            peak_count = 0
            for j in indices:
                if rotz[j][0]>thres:
                    peak_count+=1 
            pcount.append(peak_count)
            if(k==3):
                if peak_count>=20:
                    flag = 2
                else:
                    flag = 1
            if(flag==1 and k==4):
                if peak_count>=14:
                    flag = 2
                else:
                    flag = 1  
        plt.close('all')
        m = sum(pcount)
        m = m//64
        if(m==1):
            flag = 0
        Total_Images_count += 1
        if(flag==2):
            Double_compressed += 1
        elif(flag==1):
            Single_compressed += 1
        else:
            Uncompressed += 1

    print('\n\nRESULTS:')
    print('\tTotal No. of Images:',Total_Images_count)
    print('\tDouble Compressed:',Double_compressed)
    print('\tSingle Compressed:',Single_compressed)
    print('\tUncompressed:',Uncompressed)

#detect('../ucid')

#detect('../single_comp')

detect('../doub_comp/f1')
detect('../doub_comp/f2')
detect('../doub_comp/f3')
detect('../doub_comp/f4')
detect('../doub_comp/f5')
detect('../doub_comp/f6')
detect('../doub_comp/f7')
detect('../doub_comp/f8')
detect('../doub_comp/f9')
detect('../doub_comp/f10')
detect('../doub_comp/f11')
detect('../doub_comp/f12')

#detect('../test_data')

