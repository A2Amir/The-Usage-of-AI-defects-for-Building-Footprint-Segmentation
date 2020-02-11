"""This script is to train the maskRCNN model
Given the input data (Satellite images, masks and ), train an instance segmentation model to detect sustainable farming.
"""

import os
import numpy as np
import pickle
import argparse
import time

from mrcnn_config import modelConfig as MrcnnConfig
from mrcnn_config import inputConfig, modelConfig
from mrcnn_dataset import LolDataset 
import model as modellib
from config import Config
parser = argparse.ArgumentParser(description='mask RCNN')
parser.add_argument('--train', 
                    help='the directory of the training jpg dataset')

parser.add_argument('--valid', 
                    help='the directory of the validation jpg dataset')

parser.add_argument('--PretrainedModel',
                    help='the adress  of pretrained Model')

parser.add_argument('--epoch', type=int,
                    help='the number of epoches')


args = parser.parse_args()

if not args.train:
	raise ImportError('The --data parameter needs to be provided (directory to test dataset)')
else:
	train_dir = args.train

if not args.valid:
	raise ImportError('The --data parameter needs to be provided (directory to test dataset)')
else:
	val_dir = args.valid

if not args.PretrainedModel:
	raise ImportError('The --PretrainedModel parameter needs to be provided')
else:
    COCO_MODEL_PATH = args.PretrainedModel

if not args.epoch:
	raise ImportError('The --epoch parameter needs to be provided')
else:
    EPOCHES = args.epoch




output_dir = inputConfig.OUTPUT_DIR
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok = True)
    
print("############################################################################")
print ('the output will be saved in the folder %s under current directory'% inputConfig.OUTPUT_DIR)
print("############################################################################")

 
os.environ["CUDA_VISIBLE_DEVICES"]="2" # select one GPU

# Directory to save logs and trained model
MODEL_DIR = output_dir

# Path to COCO trained weights


COCO_MODEL_PATH ="/home/amir/Networke/MASK_RCNN/segmentation-maskrcnn-master/new/PretrainedModel/mask_rcnn_amir_0098.h5"

# maskRCNN config
mConfig = MrcnnConfig()
train_dir="/home/amir/DATASET/GBDX_DATA/patches/train/"
#train_dir="/home/amir/DATASET/MASK_RCNN_OHNE_JSON/train/"

print ('read training data for maskRCNN ...')
dataset_train = LolDataset()
dataset_train.load_LOL(train_dir)
dataset_train.prepare()

val_dir="/home/amir/DATASET/GBDX_DATA/patches/val/"
#val_dir="/home/amir/DATASET/MASK_RCNN_OHNE_JSON/vali/"

print ('read validating data for maskRCNN ...')
dataset_val = LolDataset()
dataset_val.load_LOL(val_dir)
dataset_val.prepare()

# Create model in training mode# Create 
model = modellib.MaskRCNN(mode="training", config=modelConfig(),
                          model_dir=MODEL_DIR)




# Which weights to start with?
#init_with = "coco"  # imagenet, coco, or last

#if init_with == "imagenet":
#    model.load_weights(model.get_imagenet_weights(), by_name=True)
#elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
model.load_weights(COCO_MODEL_PATH, by_name=True)
#elif init_with == "last":
    # Load the last model you trained and continue training
#    model.load_weights(model.find_last()[1], by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=Config.LEARNING_RATE, 
            epochs=22, 
            layers='4+')
