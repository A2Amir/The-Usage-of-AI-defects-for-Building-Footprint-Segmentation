#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:18:32 2019

@author: amir
"""
import tensorflow as tf


path_to_events_file="/media/immopixel/Amir/server/Netzwerk/MASK RCNN/new_1/metrics/amir20190620T1521/events.out.tfevents.1561036939.immopixel-System-Product-Name"
j=1
i=1
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



A=plt.plot(val_loss[:,0], val_loss[:,1], label='validation loss')
plt.plot(train_loss[:,0], train_loss[:,1], label='training loss')

plt.xlabel("Steps")
plt.ylabel("losses")
plt.title("validation Progress")
plt.legend(loc='upper right', frameon=True)

plt.savefig('/media/immopixel/Amir/server/Netzwerk/MASK RCNN/new_1/metrics/amir20190620T1521/losses.png')
plt.show()

