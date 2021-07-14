#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 21:38:22 2021

@author: pk
"""
import numpy as np
import glob
import cv2

def load_data(path):
    cv_img = []
    for img in glob.glob(path+"/*.jpg"):
        n= cv2.imread(img)
        n=n[:,:,0];n=np.expand_dims(n,0)
        cv_img.append(n)
    return cv_img
    
    


#from matplotlib import pyplot as plt
#
#plt.imshow(n, interpolation='nearest')#cmap='gray')
#plt.show()