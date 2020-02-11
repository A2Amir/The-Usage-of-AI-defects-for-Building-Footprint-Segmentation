from config import Config

class inputConfig():
    NUM_CLASSES = 1
    CLASS_DICT = {1: 'Building'}
    CATEGORIES = list(CLASS_DICT.values()) #['Building']
    CATEGORIES_VALUES = list(CLASS_DICT.keys())  #[1]
    NUM_EPOCHES = 5
    # TRAIN_LAYERS = 'all'
    # SAVE_TRAIN = 'logs'
    IMAGE_HEIGHT = 360
    IMAGE_WIDTH = 480
    JPG_NAME = 'jpg4'
    OUTPUT_DIR = "/media/immopixel/Amir/server/Netzwerk/MASK RCNN/new_1/metrics"
    MODEL_DIR ="/media/immopixel/Amir/server/Netzwerk/MASK RCNN/new_1/metrics"

class modelConfig(Config):

    # Give the configuration a recognizable name
    NAME = "Amir_ALLTamam160"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    #320
    #Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling.
    #For example, use 256, 320, 384, 448, 512, ... etc. 

    # Number of classes (including background)
    NUM_CLASSES = 1 + inputConfig.NUM_CLASSES

class inferenceConfig(modelConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3