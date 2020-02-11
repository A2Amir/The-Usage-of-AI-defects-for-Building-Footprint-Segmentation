# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:02:56 2019

@author: ziaeeamir
"""

from skimage import measure
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage.color import rgb2gray
import cv2
from config import Config

def Count_Object(Image_Adress):
    #print("Image_Adress",Image_Adress)

    # Load an color image in grayscale
    im = cv2.imread(Image_Adress,0)
   
    #plt.imshow(im)
    # cv2.waitKey(0)
    
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(im,kernel,iterations = 1)
    # cv2.imshow("erosion",erosion)
    #cv2.waitKey(0)
    #cv2.imwrite( "E:/Dataset/tirol150m/Tirol(103).png", erosion )


    blur = cv2.GaussianBlur(erosion, (5, 5), 0)
    (t, maskLayer) = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #plt.imshow(maskLayer)



    kernel = np.ones((1,1),np.uint8)
    dilation = cv2.dilate(maskLayer,kernel,iterations =1)
    #plt.imshow(dilation)
    #cv2.imwrite( "E:/Dataset/tirol150m/Tirol(103_1).png", dilation )


    blobs_labels = measure.label(dilation, background=0)
    Number_of_Object=len(np.unique(blobs_labels))

    #print("number of objects",Number_of_Object)
    #plt.imshow(blobs_labels)
    #cv2.imwrite( "E:/Dataset/tirol150m/Tirol(blobs_labels).png", blobs_labels )
    #cv2.destroyAllWindows()
    #print("Number_of_Object",Number_of_Object)
        
    return blobs_labels,Number_of_Object
    

def Show_sprate_GT(Image,Number_of_Object):

    confidence=Config.Confidence_rate

    Valid_Number_of_Object=0
    for i in range(Number_of_Object):
        number_of_objects=np.count_nonzero(Image==i)
        #print("Ground Truth NUmber %d, and the number of Pixels is %d"%(i,number_of_objects))
        if (number_of_objects>confidence):
            Valid_Number_of_Object +=1
            
    shape=(Image.shape[0],Image.shape[1],Valid_Number_of_Object-1)
    mask = np.zeros(shape, dtype="uint8")
    j=0
    i=1
    while (i<Number_of_Object):
        number_of_object=np.count_nonzero(Image==i)
        #print("Ground Truth NUmber %d, and the number of Pixels is %d"%(i,number_of_object))
        if (number_of_object>confidence):
            #a=np.expand_dims(Image==i,axis=2)
            #plt.imshow(a[:,:,0])
            mask[:,:,j:j+1]=np.expand_dims(Image==i,axis=2)
            j +=1
            #plt.title(str(j))
            #plt.pause(1)
        i +=1


    return mask

'''
img=cv2.imread("/media/amir/Amirhdd/Dataset/MASK_RCNN/Tirol/train2/polygon/3551.png",0)
if img.max() == img.min():
    print("A")

Image,Number_of_Object= Count_Object("/media/immopixel/Amir/server/dataset/MASK_RCNN/GBDX_New/dataset/Validation/labels/validation/0_copenhagen_epsg_32633_med_patch_0_480.png")
mask=Show_sprate_GT(Image,Number_of_Object)
plt.imshow(mask[:,:,2])
mask.shape[2]
plt.imshow(Image==1)
np.count_nonzero(Image==1)
['/media/amir/Amirhdd/Dataset/GBDX_DATASET/patches/val/polygon/0_copenhagen_epsg_32633_med_patch_360_0.png']
class_ids [1]
mask.shape2 (360, 480, 4)
img=mask[:,:,]

'''