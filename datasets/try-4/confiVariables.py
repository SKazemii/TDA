import os
import datetime
import numpy as np
import test as ts

CLASSES    = 2
IMG_HIGHT  = 60
IMG_WEIGHT = 40
IMG_SIZE   = (IMG_HIGHT, IMG_WEIGHT)
COLOR_TYPE = 3
EPOCHS     = 100
BATCHES    = 32
TEST_SIZE  = 10
K_FOLDS    = 10
SEED       = 123


TRAINING_DIR = "/Users/saeedkazemi/Desktop/temp/OSDN-1/casis/Dataset/data/train"

# include_top = {True | False}
include_top = False


# already done = mobilenet | inceptionresnetv2 | xception | inceptionv3 | resnet50 | vgg16 | vgg19
model_name = "vgg16"


# cross-validation
# outer_shuffle = {True | False}
outer_shuffle = True
# outer_n_splits = {3, 5, 10}
outer_n_splits = 10

# inner_shuffle = {True | False}
inner_shuffle = True
# inner_n_splits = {3, 5, 10}
inner_n_splits = 3

# GridSearchCV
# refit = {True | False}
Grid_refit = True
# Grid_n_jobs = {1:4, -1}
Grid_n_jobs = 3






# creating output paths
train_path = "/Users/saeedkazemi/Desktop/temp/OSDN-1/casis/Dataset/data/train"
test_path  = "/Users/saeedkazemi/Desktop/temp/OSDN-1/casis/Dataset/data/test"



project_path = os.getcwd()

