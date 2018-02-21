#!/usr/bin/env python3

import os
import time
import numpy as np
import random
import pickle

# matplotlib inline
import matplotlib.pyplot as plt
# OpenCV packages
# normal installation routine


import cv2
# for reading .mat files
import scipy.io as spio


# Downlaod BSDS500 image segmentation dataset from Computer Vision - Berkley group, decompress it and remove .tgz file.
tic1 = time.clock()
if not os.path.isdir("./BSR"):
    print('Connecting and downloading Texture Database from Ponce Group...')
    os.system('wget http://157.253.63.7/textures.tar.gz')
    os.system('tar -zxvf BSR_bsds500.tgz')
    os.system('rm BSR_bsds500.tgz')
else:
    print('BSDS500 dataset is already downloaded')
