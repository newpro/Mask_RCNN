
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
#import matplotlib
#import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# ## Run Object Detection


from glob import glob

# Read folder list
IMG_BASE = '/home/kits-adm/Datasets/flickr_modifiers/pics/'
LOG_BASE = '/home/kits-adm/Datasets/flickr_modifiers/filters/'
RECORD_BASE = os.path.join(LOG_BASE, 'records/')
# MASK_BASE = os.path.join(LOG_BASE, 'masks/')
ROI_BASE = os.path.join(LOG_BASE, 'rois/')
FAILURE_LOG = os.path.join(LOG_BASE, 'failures.txt')

def _log_one_line(file_loc, line):
    with open(file_loc, 'a+') as log_file:
        log_file.write(line + '\n')

# with open('/home/kits-adm/Datasets/flickr_epa/small_scale/random_selected.txt') as f:
#     lines = f.readlines()

folders = [x for x in glob(IMG_BASE + "*")]
# print (folders)

_count = 0
for folder in folders:
#     folder = (IMG_BASE + f.strip() + "/")
    print (folder, '{}%'.format(_count*100/len(folders)))
    for file in os.listdir(folder):
        if file.endswith(".jpg"):
            img_path = os.path.join(folder, file)
            category_id = img_path.split('/')[-2]
            img_id = img_path.split('/')[-1].split('.')[0]
            try:
                image = skimage.io.imread(img_path)
                results = model.detect([image])
            except: # breaks at black & white
                _log_one_line(FAILURE_LOG, img_path)
                continue
            rs = results[0]
            occur_time = 0
            for i in range(len(rs['class_ids'])):
                if rs['class_ids'][i] == 1: # human
                    roi = rs['rois'][i]
                    # note: roi x, y is flipped :<
                    _log_one_line(
                        os.path.join(ROI_BASE, ('{}.txt'.format(category_id))),
                        '{} {} {} {} {} {}'.format(img_id, occur_time, roi[1], roi[0], roi[3], roi[2]))
#                     print (rs['rois'][i])
                    occur_time += 1
            _log_one_line(os.path.join(RECORD_BASE, ('{}.txt'.format(category_id))),
                         '{} {}'.format(img_id, occur_time))
    _count += 1
print ('------ALL COMPLETED---------')
#             print (occur_time, img_path)
