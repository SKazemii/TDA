import numpy as np
# import pandas as pd
import sys, os, glob
import json, pprint
# from scipy.io import loadmat, savemat
import shutil
from matplotlib import pyplot as plt


import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import backend as K


# from keras.callbacks import ModelCheckpoint



with open("./code/config_MNIST.json") as config_file:
    config_data = json.load(config_file)




print("[INFO] tensorflow version: ", tf.version.VERSION)

"""
## Prepare the data
"""

# Model / data parameters
num_classes = config_data["dataset"]["num_classes"]
imposter_classes = config_data["dataset"]["imposter_classes"]
input_shape = (config_data["dataset"]["img_hight"], 
                config_data["dataset"]["img_weight"], 
                config_data["dataset"]["img_chanel"])

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


from gtda.images import Binarizer, RadialFiltration, ImageToPointCloud

# Pick out index of first 8 image
# Reshape to (n_samples, n_pixels_x, n_pixels_y) format


im8 = x_train



binarizer = Binarizer(threshold=0.4)
im8_binarized = binarizer.fit_transform(im8)


radial_filtration = RadialFiltration(center=np.array([6,6]))
im8_filtration = radial_filtration.fit_transform(im8_binarized)



from gtda.homology import CubicalPersistence


cubical_persistence = CubicalPersistence(n_jobs=-1)
im8_cubical = cubical_persistence.fit_transform(im8_filtration)

cubical_persistence.plot(im8_cubical)



print("finished")
sys.exit()