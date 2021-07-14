from __future__ import print_function
import keras, glob, cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential,load_model
# from keras import backend as K


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import backend as K

def load_data(path):
    cv_img = []
    for img in glob.glob(path+"/*.jpg"):
        n= cv2.imread(img)
        n=n[:,:,0];n=np.expand_dims(n,0)
        cv_img.append(n)
    return cv_img


batch_size = 128
num_classes = 2
epochs = 150

# input image dimensions
img_rows, img_cols = 60, 40


U_matrix=[]
for user in range(1,50):
    X_out=[];X_os=[]
    for o in range(1,50):
        # the data, split between train and test sets
        if o==user:
            path='/Users/saeedkazemi/Documents/Python/MasterThesis/datasets/CASIA/categorical_imgs/'+str(user)+'_subject.0/'
            X = load_data(path) 
            X = np.vstack(X)

            
        else:
            path='/Users/saeedkazemi/Documents/Python/MasterThesis/datasets/CASIA/categorical_imgs/'+str(o)+'_subject.0/'
            temp = load_data(path)
            temp=np.vstack(temp)
            X_out.append(temp)
            
    
    X_out=np.vstack(X_out) 
    
    X_out=random.sample(list(X_out),60)
    
    X_out=[np.expand_dims(i,axis=0) for i in X_out]
    X_out=np.vstack(X_out) 
   
    X_out = X_out.reshape(X_out.shape[0], img_rows, img_cols, 1)
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
 
    print((X_out.shape))
    print((X.shape))
    1/0
    input_shape = (img_rows, img_cols, 1)

    X = X.astype('float32')
    X_out = X_out.astype('float32')
    X /= 255
    X_out /= 255

    print('x_train shape:', X.shape)
    print(X.shape[0], 'User samples')
    print(X_out.shape[0], 'Out samples')
    


    # convert class vectors to binary class matrices
    Y = np.ones(len(X))
    Y_out = np.zeros(len(X_out))
    Y_total=np.append(Y,Y_out)
    X_total=np.concatenate((X,X_out))
    

    x_train, x_test, y_train, y_test = train_test_split(X_total,Y_total,test_size=0.2,train_size=0.8)
    x_train, x_cv, y_train, y_cv = train_test_split(x_train,y_train,test_size = 0.25,train_size =0.75)
    
    
    ####################################        
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    #model.summary()
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    weight_path="{}_weights.best.hdf5".format('casia-D_classification')
        
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = False)
    # history=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_cv, y_cv),callbacks = [checkpoint],shuffle=True)
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    model1=load_model('/Users/saeedkazemi/Documents/Python/MasterThesis/casia-D_classification_weights.best.hdf5')
    score_cv = model1.evaluate(x_cv, y_cv, verbose=0)
    score_te = model1.evaluate(x_test, y_test, verbose=0)


    ##########################ok
    for os in range(51,98):#opens set
        path='/Users/saeedkazemi/Documents/Python/MasterThesis/datasets/CASIA/categorical_imgs/'+str(os)+'_subject.0/'
        temp = load_data(path);temp=np.vstack(temp);X_os.append(temp)
    X_os=np.vstack(X_os); X_os = X_os.reshape(X_os.shape[0], img_rows, img_cols, 1);X_os = X_os.astype('float32');  X_os /= 255
    y_os=np.zeros(len(X_os))
    score_os = model1.evaluate(X_os, y_os, verbose=0)
        
    U_matrix.append([user,score_cv[1],score_te[1],score_os[1]])
    #K.clear_session()
#    print('Test loss:', score[0])
#    print('Test accuracy:', score[1])
np.savetxt('Casia=D_results.csv',np.vstack(U_matrix),delimiter=',')
