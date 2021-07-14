# filter warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, Dense, Flatten
from keras.utils.np_utils import to_categorical

from keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    EarlyStopping,
    ReduceLROnPlateau,
)
# other imports
from sklearn import preprocessing as prepro
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

import numpy as np
import pandas as pd
import glob, sys, os
import cv2
import datetime
import time, re
import pickle
from scipy.io import savemat



os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
tf.config.experimental.disable_mlir_graph_optimization()

# importing setting
import confiVariables as cfg

# check if CUDA is available
if tf.config.list_physical_devices('GPU'):
     print('CUDA is available!  Training on GPU ...')


# reading dataset
df_tr = pd.read_csv("casis/Dataset/data/train/df.csv")

X = list()

for i in range(df_tr.shape[0]):
    df_tr.loc[i, "label1"] = "1" if df_tr.loc[i,"label"]== 4 else "0" 
    img_arr = cv2.imread(df_tr.loc[i, "path"], cv2.IMREAD_COLOR)
    # img_arr = cv2.resize(img_arr, cfg.IMG_SIZE)
    img_arr = preprocess_input(img_arr)
    X.append(img_arr)

y = df_tr["label1"].values
y = to_categorical(y, 2)

X = np.array(X).reshape((-1, cfg.IMG_HIGHT, cfg.IMG_WEIGHT, 3))


# train_datagen = image.ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
# train_batches = train_datagen.flow(X, y, batch_size=cfg.BATCHES)


# cv2.imshow("output", X[1,:,:,:])
# cv2.waitKey(1000)
# print(y)
# print(df_tr.head(30))
# sys.exit()




# create the pretrained models
if cfg.model_name == "vgg16":
    image_size = (cfg.IMG_HIGHT, cfg.IMG_WEIGHT, cfg.COLOR_TYPE)
    base_model = VGG16(weights="imagenet", input_shape=image_size, include_top=False)


    # Freeze all layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(cfg.CLASSES, activation = 'softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # print(model.summary())    
else:
    base_model = None

print("[INFO] successfully loaded base model and model...")




checkpoint = [
        tf.keras.callbacks.ModelCheckpoint(
            "./casis/vgg16-casia.hdf5", save_best_only=True, monitor="val_loss"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]    


history = model.fit(
    X,
    y,
    batch_size=cfg.BATCHES,
    callbacks=[checkpoint],
    epochs=cfg.EPOCHS,
    validation_split=0.1,
    verbose=1,
)

