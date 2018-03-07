import os
import time
import numpy as np
import random
import pickle
import glob

# matplotlib inline
import matplotlib.pyplot as plt
# OpenCV packages
# normal installation routine


import cv2
# for reading .mat files
import scipy.io as spio
from skimage import io, color


a = spio.loadmat('../BSDS_tiny/'+'24063.mat')['groundTruth']


segm = a[0,[1]][0]['Segmentation'][0,0]
bound = a[0,[0]][0]['Boundaries'][0,0]

print(np.unique(segm))


plt.imshow(np.squeeze(segm), cmap=plt.get_cmap('summer'))
plt.show()