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




if not os.path.isfile("./train_test_ims.pickle"):

    from read_n_download_dataset import read_n_download_dataset

    im_train , im_test = read_n_download_dataset()
    data_dict = {'Train_Images': im_train, 'Test_Images': im_test}
    with open('./train_test_ims.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    data = pickle.load( open( "./train_test_ims.pickle", "rb" ) )
    im_train  = data['Train_Images']
    im_test  = data['Test_Images']


# Append the function provided in lib
sys.path.append('../lib/python')

#Create a filter bank with deafult params
from fbCreate import fbCreate

# return teh bank of filters with 8 orientations and 2 scales, it would be nice to add
# visualization of the bank of filters.
fb = fbCreate()


#Load sample images from disk
from skimage import color
from skimage import io

#Set number of clusters
k = 25

#Apply filterbank to sample image
from fbRun import fbRun


im_train = np.reshape(im_train,(256,256,-1))
ims_to_textons = im_train[:,:,0]

print("This is the shape of the images to train {}".format(im_train.shape))

for i in range(1,im_train.shape[2]):
    im1 =  np.squeeze(ims_to_textons)
    print(im1.shape)
    im2 = np.squeeze(im_train[:,:,i])
    print(im2.shape)

    ims_to_textons = np.hstack( (im1 , im2)  )

print("This is the images sample shape {}".format(ims_to_textons.shape))






#Load more images
imTest1=color.rgb2gray(io.imread('../img/person2.bmp'))
imTest2=color.rgb2gray(io.imread('../img/goat2.bmp'))

#Calculate texton representation with current texton dictionary
from assignTextons import assignTextons

tmapBase1 = assignTextons( fbRun(fb,imBase1),textons.transpose() )
tmapBase2 = assignTextons( fbRun(fb,imBase2),textons.transpose() )
tmapTest1 = assignTextons( fbRun(fb,imTest1),textons.transpose() )
tmapTest2 = assignTextons( fbRun(fb,imTest2),textons.transpose() )

#Check the euclidean distances between the histograms and convince yourself that 
#the images of the goats are closer because they have similar texture pattern

# --> Can you tell why we need to create a histogram before measuring the distance? <---

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

D = np.linalg.norm(histc(tmapBase1.flatten(), np.arange(k))/tmapBase1.size - \
     histc(tmapTest1.flatten(), np.arange(k))/tmapTest1.size)
print('Similaridad entre humanos {}'.format(D))

D = np.linalg.norm(histc(tmapBase1.flatten(), np.arange(k))/tmapBase1.size - \
     histc(tmapTest2.flatten(), np.arange(k))/tmapTest2.size)
print('Similaridad entre imagenes {}'.format(D))


D = np.linalg.norm(histc(tmapBase2.flatten(), np.arange(k))/tmapBase2.size - \
     histc(tmapTest1.flatten(), np.arange(k))/tmapTest1.size)
print('Similaridad entre imagenes {}'.format(D))


D = np.linalg.norm(histc(tmapTest2.flatten(), np.arange(k))/tmapTest2.size - \
     histc(tmapBase2.flatten(), np.arange(k))/tmapBase2.size)
print('Similaridad entre cabras {}'.format(D))



