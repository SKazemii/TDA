import numpy as np
import pandas as pd
import sys, os, glob
import json, pprint
from scipy.io import loadmat, savemat
import shutil
import random

from tqdm import tqdm

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K


from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


from packages import compute_openmax, evt_fitting, MAV_Compute, compute_distances




def load_data(path):
    imgs = list()
    for img in glob.glob(path + "/*.jpg"):
        temp = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        temp = np.expand_dims(temp, 0)
        imgs.append(temp)
    imgs = np.vstack(imgs)
    return imgs





with open("./code/config.json") as config_file:
    config_data = json.load(config_file)




print("[INFO] tensorflow version: ", tf.version.VERSION)

"""
## Prepare the data
"""

# Model / data parameters
num_classes = config_data["dataset"]["num_classes"]
imposter_classes = config_data["dataset"]["imposter_classes"]
input_shape = [config_data["dataset"]["img_hight"], 
                config_data["dataset"]["img_weight"], 
                config_data["dataset"]["img_chanel"]]

# the data, split between train and test sets

dataset1 = list()
Y_subjects = list()
X_subjects = list()
for subject in range(1,97):
    path = '/Users/saeedkazemi/Documents/Python/MasterThesis/datasets/CASIA/categorical_imgs/' + str(subject) + '_subject.0/'
    X_subjects.append(load_data(path))
    temp = X_subjects[subject-1].shape[0]
    Y_subjects.append(subject*np.ones((temp,1)))

X_subjects = np.vstack(X_subjects)
Y_subjects = np.vstack(Y_subjects)
print(X_subjects.shape)
print(Y_subjects)


# test_ind = np.where(np.logical_and(Y_subjects>= 2, Y_subjects<= 50))
test_ind = np.where(Y_subjects = 55)
y_test = Y_subjects[test_ind[0]]
print(y_test)


sys.exit()


X_user = list()
X_other = list()
user = 1
for other in range(1,50):
    if other == user:
        path = '/Users/saeedkazemi/Documents/Python/MasterThesis/datasets/CASIA/categorical_imgs/' + str(user) + '_subject.0/'
        X_user = load_data(path) 
    else:
        path = '/Users/saeedkazemi/Documents/Python/MasterThesis/datasets/CASIA/categorical_imgs/' + str(other) + '_subject.0/'
        temp = load_data(path)
        X_other.append(temp)


X_other = np.vstack(X_other)   
X_other = random.sample(list(X_other), 60)
X_other = [np.expand_dims(i, axis=0) for i in X_other]
X_other = np.vstack(X_other) 


# Make sure images have shape (28, 28, 1)
X_user = np.expand_dims(X_user, -1)
X_other = np.expand_dims(X_other, -1)


# Scale images to the [0, 1] range
X_user = X_user.astype("float32") / 255
X_other = X_other.astype("float32") / 255


# convert class vectors to binary class matrices
Y_user  = np.ones(len(X_user))
Y_other = np.zeros(len(X_other))

YY = np.append(Y_user, Y_other)
XX = np.concatenate((X_user, X_other))


#TODO
x_train, x_test, y_train, y_test = train_test_split(XX, YY, test_size = 0.2, train_size = 0.8)
# x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size = 0.25, train_size = 0.75)




print("[INFO] x_train shape:", x_train.shape)
print("[INFO]", x_train.shape[0], "train samples")
print("[INFO]", x_test.shape[0], "test samples")
# print("[INFO]", x_cv.shape[0], "Validation samples")



# train_ind = np.where(y_train < num_classes)

# x_train = x_train[train_ind[0], :, :, :]
# y_train = y_train[train_ind[0]]
y_tr = y_train


# test_ind_2 = np.where(y_test < num_classes)

# x_test_2 = x_test[test_ind_2[0], :, :, :]
# y_test_2 = y_test[test_ind_2[0]]
y_te = y_test


print("[INFO] x_train shape:", x_train.shape)
print("[INFO] y_train shape:", y_train.shape)
print("\n\n")


# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

    
# print(y_test)

# sys.exit()


"""
## Build and Train the model
"""

if config_data["convnet"]["train_mode"] == True:
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),

            layers.Dense(num_classes),
            layers.Softmax()
        ]
    )

    model.summary()
    optimizer = tf.optimizers.Adadelta()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        
    


    checkpoint = ModelCheckpoint(
        config_data["convnet"]["path_saved_model"] + str(user) + ".hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=False,
    )

    


    history = model.fit(
        x_train,
        y_train,
        verbose=1,
        batch_size=config_data["convnet"]["batch_size"],
        validation_split=config_data["convnet"]["validation_split"],  #validation_data=(x_cv, y_cv)
        callbacks=[checkpoint],
        epochs=config_data["convnet"]["epochs"],
        shuffle=True,
    )


saved_model = tf.keras.models.load_model(config_data["convnet"]["path_saved_model"]+ str(user) + ".hdf5" )



"""
## Evaluate the trained model
"""

if config_data["convnet"]["Evaluate"] == True:

    print("\n\n############################")
    score = saved_model.evaluate(x_test, y_test, verbose=2)
    print("[INFO] Test loss: {:.2f} %".format( score[0]))
    print("[INFO] Test accuracy: {:.2f} %".format( 100*score[1]))


    print("\n\n############################")
    score = saved_model.evaluate(x_train, y_train, verbose=1)
    print("[INFO] Train loss: {:.2f} %".format( score[0]))
    print("[INFO] Train accuracy: {:.2f} %".format( 100*score[1]))



inp = saved_model.input 
outputs = [layer.output for layer in saved_model.layers[-3:]]           # all layer outputs
functors = [K.function([inp], [out]) for out in outputs]                # evaluation functions




if config_data["convnet"]["compute_trained_features"] == True:

    path = config_data["path"]["path_saved_outputs"] 
    if not os.path.exists(path):
        os.makedirs(path)

    print("\n\n############################")
    for i in tqdm(range(x_train.shape[0]), desc="[INFO] Calculating features ... "):

        X = K.constant(np.expand_dims(x_train[i,:,:,:], 0))
        
        layer_outs = [func([X]) for func in functors]
        # print(layer_outs)
        # sys.exit()
        data = {"fc1": layer_outs[0][0],
                "fc2": layer_outs[1][0],
                "score": layer_outs[2][0],
                "X" : x_train[i,:,:,:],
                "y" : y_train[i],
                "y_cat" : y_tr[i],
        }

        path = config_data["path"]["path_saved_outputs"] + "class_{}".format(int(y_tr[i]))
        if not os.path.exists(path):
            os.makedirs(path)



        file_name = config_data["path"]["path_saved_outputs"] + "class_{}/image_{}.mat".format(int(y_tr[i]), i)
        savemat(file_name, data)


    print("[INFO] Trained features were completed!")





    path = config_data["path"]["path_dist_outputs"]
    if not os.path.exists(path):
        os.makedirs(path) 



    labellist = ["class_{}".format(i) for i in range(num_classes)]
    
    print("\n\n############################")
    for category_name in labellist:
        print("[INFO] Computing MAV for: ", category_name)
        MAV_Compute.compute_mean_vector(category_name)
    
    
    
    for category_name in labellist:
        mav_fname = config_data["path"]["path_MAV_outputs"] + category_name + ".mat"
        feature_path = config_data["path"]["path_saved_outputs"] + category_name + "/"

        distance_distribution = compute_distances.compute_distances( mav_fname, category_name, feature_path )
        savemat(path + "{}_distances.mat".format(category_name), distance_distribution)

    print("[INFO] MAV and distances were completed!")






if config_data["convnet"]["compute_tested_features"] == True:

    test_ind = np.where(y_test < num_classes + imposter_classes)

    x_test = x_test[test_ind[0], :, :, :]
    y_test = y_test[test_ind[0]]
    y_te = y_test


    y_test = keras.utils.to_categorical(y_test, num_classes + imposter_classes)


    path = config_data["path"]["path_test_outputs"]
    
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


    b = 200 #x_test.shape[0]
    print("\n\n############################")
    acc = np.zeros((b,5))#x_test.shape[0]
    for i in tqdm(range (b), desc="[INFO] Calculating test features ... ", ncols=100):#x_test.shape[0]

        X = K.constant(np.expand_dims(x_test[i,:,:,:], 0))
        
        layer_outs = [func([X]) for func in functors]

        data = {"fc1": layer_outs[0][0],
                "fc2": layer_outs[1][0],
                "score": layer_outs[2][0],
                "y" : y_test[i],
                "y_cat" : y_te[i],
        }


        distance_path = config_data["path"]["path_dist_outputs"]
        mean_path = config_data["path"]["path_MAV_outputs"]
        alpha_rank = config_data["OpenMax"]["ALPHA_RANK"]
        weibull_tailsize = config_data["OpenMax"]["WEIBULL_TAIL_SIZE"]

        labellist = ["class_{}".format(i) for i in range(num_classes)]
        
        weibull_model = evt_fitting.weibull_tailfitting(mean_path, distance_path, labellist,
                                            tailsize = weibull_tailsize, distance_type = 'eucos')

        
        

        openmax, softmax =  compute_openmax.recalibrate_scores(weibull_model, labellist, data)



        if y_te[i] < config_data["dataset"]["num_classes"]:
            a = y_te[i]
        else:
            a = config_data["dataset"]["num_classes"]
        # print("Image class: {}".format(a))
        # print("Softmax Scores ", softmax.argmax())
        # print("Openmax Scores ", openmax.argmax())
        # print(openmax.shape, softmax.shape)

        acc[i, 0] = a
        acc[i, 1] = softmax.argmax()
        acc[i, 2] = openmax.argmax()

        if softmax[softmax.argmax()] < config_data["OpenMax"]["THRESHOLD"]:
            acc[i, 3] = softmax.argmax()
        else:
            acc[i, 3] = config_data["dataset"]["num_classes"]
            
        if openmax[openmax.argmax()] < config_data["OpenMax"]["THRESHOLD"]:
            acc[i, 4] = openmax.argmax()
        else:
            acc[i, 4] = config_data["dataset"]["num_classes"]

        data1 = {"fc1": layer_outs[0][0],
                "fc2": layer_outs[1][0],
                "score": layer_outs[2][0],
                "X" : x_test[i,:,:,:],
                "y" : y_test[i],
                "y_cat" : y_te[i],
                "Softmax_Scores": softmax,
                "Openmax_Scores": openmax,
                "Softmax_argmax": softmax.argmax(),
                "Openmax_argmax": openmax.argmax(),
        }


        file_name = config_data["path"]["path_test_outputs"] + "{}_Image_{}.mat".format(i, y_te[i])
        savemat(file_name, data1)

        # pprint.pprint(data)
        # if i == 200: 
        #     exit()

    print("[INFO] Tested features were completed!")
    df = pd.DataFrame(acc, columns=["clss", "softmax", "openmax", "soft", "open"])
    if os.path.exists(config_data["path"]["path_csv"]):
        os.remove(config_data["path"]["path_csv"])
    df.to_csv(config_data["path"]["path_csv"])



# pprint.pprint(data1)
print(df.head(30))
df = pd.read_csv(config_data["path"]["path_csv"])
print("\n\n############################")
print("Total samples:   {}".format(df.shape[0]))
print("Unknown samples: {}, {:.2f} %".format(df.loc[ df["clss"] >= num_classes ].shape[0], 100*df.loc[ df["clss"] >= num_classes ].shape[0]/df.shape[0]))
print("Known samples:   {}, {:.2f} %".format(df.loc[ df["clss"] <  num_classes ].shape[0], 100*df.loc[ df["clss"] < num_classes ].shape[0]/df.shape[0]))


print("\n\n############################")
print("Accuracy with unknown samples")
print("[INFO] Softmax accuracy: {:.2f} %".format( 100*(((df["clss"]==df["softmax"])*1).sum()/df.shape[0])))
print("[INFO] Openmax accuracy: {:.2f} %".format( 100*(((df["clss"]==df["openmax"])*1).sum()/df.shape[0])))

print("[INFO] Softmax+threshold accuracy: {:.2f} %".format( 100*(((df["clss"]==df["soft"])*1).sum()/df.shape[0])))
print("[INFO] Openmax+threshold accuracy: {:.2f} %".format( 100*(((df["clss"]==df["open"])*1).sum()/df.shape[0])))


df2 = df.loc[ df["clss"] < num_classes ]
print("\n\n############################")
print("Accuracy without unknown samples")
print("[INFO] Softmax accuracy: {:.2f} %".format( 100*(((df2["clss"]==df2["softmax"])*1).sum()/df2.shape[0])))
print("[INFO] Openmax accuracy: {:.2f} %".format( 100*(((df2["clss"]==df2["openmax"])*1).sum()/df2.shape[0])))

print("[INFO] Softmax+threshold accuracy: {:.2f} %".format( 100*(((df2["clss"]==df2["soft"])*1).sum()/df2.shape[0])))
print("[INFO] Openmax+threshold accuracy: {:.2f} %".format( 100*(((df2["clss"]==df2["open"])*1).sum()/df2.shape[0])))


# print(df.loc[ df["clss"] != df["openmax"] ])