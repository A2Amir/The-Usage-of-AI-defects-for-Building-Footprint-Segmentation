# MaskRCNN for Instance Segmentation of Building Footprints
MASK RCNN Model is changed by Amir Ziaee

## Prerequisites
- Python 3.4+
- [Imgaug](https://github.com/aleju/imgaug)
- Tensorflow 1.3+
- Keras 2.0.8+

## Get started

(https://github.com/olgaliak/segmentation-unet-maskrcnn/blob/master/maskRCNN/main_train.py) 

## change Configuration

In config.py you can change Learning Rate and LEARNING_MOMENTUM and WEIGHT_DECAY
The size of input images you can change in mrcnn_config (IMAGE_HEIGHT,IMAGE_WIDTH)


## Input data folder structure for maskRCNN

      - disaster_app
      | - train
      | |- jpg4 (Satellite images)
      | |- polygon (masks)
      | - valid
      | |- jpg4 (Satellite images)
      | |- polygon (masks)
      | - test
      | |- jpg4 (Satellite images)
      | |- polygon (masks)      


            

## Train maskRCNN
In general, we need millions of images to train a deep learning model from scratch. As discussed before, we have unbalanced training dataset and the training dataset size is not enough for training a MaskRCNN model from scratch.

To leverage training time and the tiny training dataset, transfer learning is performed in the project. Transfer learning is a machine learning technique where a model trained from one task is repurposed on another related task. In our case, we used Feature Pyramid Network (FPN) and a ResNet101 backbone. The initial weights is located in this folder  that have been trained on  Crowd AI dataseted. These pre-trained weights already learned common features in natural images.

runing and executing :

> python main_train.py --train <'training data folder'> --valid <'validation data folder'> --PretrainedModel <'PretrainedModel data folder'> --epoch <'epoch number'>

   python main_train.py 
   --train /home/amir/DATASET/MASK_RCNN_OHNE_JSON/train/
   --valid /home/amir/DATASET/MASK_RCNN_OHNE_JSON/vali/ 
   --epoch 1 
   --PretrainedModel /home/amir/Networke/MASK_RCNN/crowdai-mapping-challenge-mask-rcnn/logs/    crowdai-mapping-challenge20181219T1054/mask_rcnn_crowdai-mapping-challenge_0040.h5



## Evaluate model (thies habe ich noch nicht angepasst)

Our primary metric for model evaluation was Jaccard Index and Dice Similarity Coefficient. They measure how close predicted mask is to the manually marked masks, ranging from 0 (no overlap) to 1 (complete congruence).

[Jaccard Similarity Index](https://en.wikipedia.org/wiki/Jaccard_index) is the most intuitive ratio between the intersection and union:

> ![](https://latex.codecogs.com/svg.latex?J%28A%2CB%29%20%3D%20%5Cfrac%7B%7CA%5Ccap%20B%7C%7D%7B%7CA%7C&plus;%7CB%7C-%7CA%5Ccap%20B%7C%7D)

[Dice Coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) is a popular metric, it's numerically less sensitive to mismatch when there is a reasonably strong overlap:

> ![](https://latex.codecogs.com/svg.latex?DSC%28A%2CB%29%20%3D%20%5Cfrac%7B2%7CA%5Ccap%20B%7C%7D%7B%7CA%7C&plus;%7CB%7C%7D)

Regarding loss functions we started out with using classical Binary Cross Entropy (BCE) that is available as prebuilt loss function in Keras.
Also we have explored incorporating Dice Similarity Coefficient  into loss function and got inspiration from [this repo](https://github.com/killthekitten/kaggle-carvana-2017) related to Kaggle's Carvana challenge. 

> ![](https://latex.codecogs.com/svg.latex?BCE_%7BDSC%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7DBCE&plus;%5Cfrac%7B1%7D%7B2%7D%7B%281-DSC%29%7D)

To get the jaccard and dice coefficient for each test image, run the script [main_eval.py](https://github.com/olgaliak/segmentation-unet-maskrcnn/blob/master/maskRCNN/main_eval.py). These values will be saved in pickle files. By turn the hill parrameter (True/False), you can choose model with/without hill shade data.

> python main_eval.py --data <'test data folder'> --hill <'True/False'> --output <'output folder'> --epoch <'a list of epoch numbers'>

> eg: python main_eval.py --data ./tstData --hill True --output ./output --epoch [50]


