from sklearn.pipeline import make_pipeline, make_union
from gtda.diagrams import PersistenceEntropy, Scaler, Amplitude
from gtda.images import HeightFiltration, RadialFiltration, Binarizer

from gtda.homology import CubicalPersistence



import numpy as np
# import pandas as pd
import sys, os, glob
import json, pprint
from scipy.io import loadmat, savemat
import shutil
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras
# from keras import layers
# from keras import backend as K


# from keras.callbacks import ModelCheckpoint




with open("./code/config_MNIST.json") as config_file:
    config_data = json.load(config_file)




print("[INFO] tensorflow version: ", tf.version.VERSION)



direction_list = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

center_list = [
    [13, 6],
    [6, 13],
    [13, 13],
    [20, 13],
    [13, 20],
    [6, 6],
    [6, 20],
    [20, 6],
    [20, 20],
]

# Creating a list of all filtration transformer, we will be applying
filtration_list = (
    [
        HeightFiltration(direction=np.array(direction), n_jobs=-1)
        for direction in direction_list
    ]
    + [RadialFiltration(center=np.array(center), n_jobs=-1) for center in center_list]
)

# Creating the diagram generation pipeline
diagram_steps = [
    [
        Binarizer(threshold=0.4, n_jobs=-1),
        filtration,
        CubicalPersistence(n_jobs=-1),
        Scaler(n_jobs=-1),
    ]
    for filtration in filtration_list
]

# Listing all metrics we want to use to extract diagram amplitudes
metric_list = [
    {"metric": "bottleneck", "metric_params": {}},
    {"metric": "wasserstein", "metric_params": {"p": 1}},
    {"metric": "wasserstein", "metric_params": {"p": 2}},
    {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 1, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 2, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {"p": 2, "n_layers": 1, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {"p": 2, "n_layers": 2, "n_bins": 100}},
    {"metric": "betti", "metric_params": {"p": 1, "n_bins": 100}},
    {"metric": "betti", "metric_params": {"p": 2, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 1, "sigma": 1.6, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 1, "sigma": 3.2, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 2, "sigma": 1.6, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 2, "sigma": 3.2, "n_bins": 100}},
]

#
feature_union = make_union(
    *[PersistenceEntropy(nan_fill_value=-1)]
    + [Amplitude(**metric, n_jobs=-1) for metric in metric_list]
)

tda_union = make_union(
    *[make_pipeline(*diagram_step, feature_union) for diagram_step in diagram_steps],
    n_jobs=-1
)





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



train_ind = np.where(y_train < num_classes)

x_train = x_train[train_ind[0], :, :]
y_train = y_train[train_ind[0]]
y_tr = y_train


test_ind_2 = np.where(y_test < num_classes)

x_test_2 = x_test[test_ind_2[0], :, :]
y_test_2 = y_test[test_ind_2[0]]
y_te_2 = y_test_2


print("[INFO] x_train shape:", x_train.shape)
print("[INFO] y_train shape:", y_train.shape)
print("\n\n")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test_2 = keras.utils.to_categorical(y_test_2, num_classes)



# sys.exit()
X_train_tda = tda_union.fit_transform(x_train[1:10,:,:])
print(X_train_tda.shape)
print(type(X_train_tda))

np.save("./code/X_train_tda.npy", X_train_tda)
np.save("./code/y_train.npy", y_train)
