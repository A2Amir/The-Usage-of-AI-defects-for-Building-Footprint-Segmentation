#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:07:18 2019

@author: amir
"""


# import maskRCNN utils
from mrcnn_config import modelConfig as MrcnnConfig
from mrcnn_config import inputConfig, inferenceConfig
from mrcnn_dataset import LolDataset 
import model as modellib
import cv2
import visualize
import argparse
import glob




parser = argparse.ArgumentParser(description='mask RCNN')
parser.add_argument('--Satellite_dir',    help='the directory of the satellite images')

parser.add_argument('--Save', 
                    help='Where data will be stored ')

parser.add_argument('--PretrainedModel',
                    help='the adress  of pretrained Model')



args = parser.parse_args()

if not args.Satellite_dir:
	raise ImportError('The --data parameter needs to be provided (directory of images)')
else:
	test_dir = args.Satellite_dir

if not args.Save:
	raise ImportError('The --data parameter needs to be provided (Directory of the folder in which the data is to be stored.)')
else:
	prediction_dir = args.Save

if not args.PretrainedModel:
	raise ImportError('The --PretrainedModel parameter needs to be provided')
else:
    COCO_MODEL_PATH = args.PretrainedModel

MODEL_PATH="/media/immopixel/Amir/server/Netzwerk/MASK RCNN/new_1/metrics/amir_alltamam16020190629T2333/mask_rcnn_amir_alltamam160_0120.h5"
test_dir="/media/immopixel/Amir/server/dataset/MASK_RCNN/GBDX_DATASET/patches_new/test/jpg4/"
prediction_dir="/media/immopixel/Amir/server/dataset/MASK_RCNN/GBDX_DATASET/patches_new/test/tamam/"


output_dir = inputConfig.OUTPUT_DIR
model_dir = inputConfig.MODEL_DIR

mConfig = MrcnnConfig()

inference_config = inferenceConfig()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=model_dir)
print("Loading weights from ", MODEL_PATH)
model.load_weights(MODEL_PATH, by_name=True)

images=glob.glob(test_dir+'*.png')

for image in images:
    name=image.split('/')[-1]
    print ('processing image_id {}'.format(name))
    # load groundtruth and prediction per image
    original_image=cv2.imread(image)
    original_image=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    results = model.detect([original_image]*3, verbose=1)
    r = results[0]
    class_names = ['BG', 'building'] 
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],class_names,r['scores'],title=str(name),figsize=(7,7),prediction_dir=prediction_dir)


