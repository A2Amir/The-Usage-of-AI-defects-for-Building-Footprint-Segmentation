#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:18:32 2019

@author: amir
"""
import tensorflow as tf
path_to_events_file="/home/amir/Networke/MASK_RCNN/segmentation-maskrcnn-master/new/metrics/amir20190301T2111/events.out.tfevents.1551471141.immopixel-MS-7A58"
i=1
j=1
val=[]
train=[]
for e in tf.train.summary_iterator(path_to_events_file):
    for v in e.summary.value:
        if v.tag == 'loss' or v.tag == 'accuracy':
            train.append([i,v.simple_value])
            i +=1
        if v.tag == 'val_loss' or v.tag == 'val_accuracy':
            val.append([j,v.simple_value])
            j +=1
            
val_loss=np.asarray(val)
train_loss=np.asarray(train)
val_loss

%matplotlib inline  

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt



plt.plot(train_loss[:,0], train_loss[:,1], label='training loss')
plt.plot(val_loss[:,0], val_loss[:,1], label='validation loss')

plt.xlabel("Steps")
plt.ylabel("losses")
plt.title("validation Progress")
plt.legend(loc='upper right', frameon=True)
plt.show()


