#!/usr/bin/env python3

import os
import time
import numpy as np
import random
import pickle

# matplotlib inline
import matplotlib.pyplot as plt
# OpenCV packages
import cv2
# for reading .mat files
import scipy.io as spio

import sys




data = pickle.load( open( "./train_test_ims.pickle", "rb" ) )
im_train  = data['Train_Images']
im_test  = data['Test_Images']

print("images size {}".format(im_train.shape))

im_test = np.reshape(im_test,(256,256,-1))
im_train = np.reshape(im_train,(256,256,-1))




plt.imshow(np.squeeze(im_train[:,:,]),cmap='gray')
plt.show()