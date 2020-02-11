import utils
import os
import cv2
import numpy as np
from mrcnn_config import inputConfig
from PIL import Image
import matplotlib.pyplot as plt
from Extract_GT import Count_Object,Show_sprate_GT
###
def generate_mask_path(mask_dir, filename):
    #fn_img, ext = os.path.splitext(os.path.basename(filename))
    #mask_endings = [x for x in inputConfig.CATEGORIES if x != fn_img.split('_')[0]]
    mask_path = [os.path.join(mask_dir, filename)]
   # for ending in mask_endings:
   #     mask_path.append( os.path.join(mask_dir, fn_img + '_'+ ending + '.jpg'))
    return mask_path

class LolDataset(utils.Dataset):
    
    def load_LOL(self, datasetdir,hill=True):
        
        for i in range(inputConfig.NUM_CLASSES):
            self.add_class("shapes", i, inputConfig.CLASS_DICT[i+1] )
        #if hill == True:
            image_dir = os.path.join(datasetdir, 'images')
        #else:
        #    image_dir = os.path.join(datasetdir, 'jpg')
        print ('image_dir is:', image_dir)
        mask_dir = os.path.join(datasetdir, 'labels')
        print ('Mask_dir is:', mask_dir)
        
        image_names_ohne_valid_mask = next(os.walk(image_dir))[2]
        print('Number of images ohne valid MASKs:',len(image_names_ohne_valid_mask))
        
        image_names=[]
        for i in range(len(image_names_ohne_valid_mask)):
            path=os.path.join(mask_dir, image_names_ohne_valid_mask[i])
            img=cv2.imread(path,0)
            #print("LOL DATASE MASK",path)
            if img.max() != img.min():
                mask0,Number_of_Object= Count_Object(path)
                mask=Show_sprate_GT(mask0,Number_of_Object)
                #print("LOL DATASE MASK",mask.shape)
                if (mask.shape[2]>=1):
                    image_names.append(image_names_ohne_valid_mask[i])
            #else:
                #print("Kein Grund Thruth")
                #print("Grund Thruth path:",path)

        print('Number of images mit valid MASKs:',len(image_names))

        valid_mask_list=datasetdir.split('/')
        #print(valid_mask_list)
        train=False
        for j in (valid_mask_list):
            if (j=="train"):
                train=True
        print("Train",train)
                
        for i in range(len(image_names)):
            self.add_image("shapes", image_id = i,
                    path=os.path.join(image_dir, image_names[i]),
                    mask_path = generate_mask_path(mask_dir, image_names[i]),
                    width = inputConfig.IMAGE_WIDTH,
                    height = inputConfig.IMAGE_HEIGHT,
                    train=train
                    )
        
    def load_image(self, image_id):
        info = self.image_info[image_id]
        image_path = info['path']
        #print("image_path",image_path)
        
        image_BGR = cv2.imread(image_path)
        #print("load_image1",image_BGR.shape)

        image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, ( info['width'],info['height'])) 

        #print("load_image2",image.shape)

        return image

    def load_mask(self, image_id):
        #global mask
        info = self.image_info[image_id]
        #print(info)
        mask_path = info['mask_path']
        valid_mask = []
        #print(mask_path)
        for _mask_path in mask_path:
            _mask = cv2.imread(_mask_path, 0)
            #print(_mask.shape)
            
            if _mask.max() == _mask.min():
                pass
            else:
                valid_mask.append(_mask_path)
                
                
                
             
        #print(valid_mask)
        shapes =list(inputConfig.CLASS_DICT.values()) #['Building']
        ##train = info['train']
        ##if (train==False):
            #print("Validation is",train)
        ##    count = len(valid_mask)
                #print("info['width'],info['height']",info['width'],info['height'])
        ##    mask = np.zeros([ info['height'],info['width'], count], 'uint8')
               #shapes = []

            
       ##     img_array = cv2.imread(valid_mask[0], 0)
                   #print("img_array1",img_array.shape)

                   #img_array = cv2.resize(img_array, ( info['width'],info['height'])) 
       
            #print("img_array2",img_array.shape)
            
        ##    (thresh, im_bw) = cv2.threshold(img_array, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ##    mask_array = (img_array > thresh).astype('uint8')
            #mask_array=Image.fromarray(thresh)

        ##   mask[:, :, 0:1] = np.expand_dims(mask_array, axis=2)
            #print("mask.shape1 of validation",mask.shape)

       ## if (train==True):
            #print("Train is",train)
        mask0,Number_of_Object= Count_Object(valid_mask[0])
        mask=Show_sprate_GT(mask0,Number_of_Object)

       
           
         #print("mask0.shape1 ",mask0.shape)
       ## mask=mask

        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s) for s in shapes])
        #print("class_ids",class_ids)
        #print("mask.shape2",mask.shape)
        return mask, class_ids