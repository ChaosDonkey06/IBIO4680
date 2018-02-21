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
if not os.path.isdir("./data_textures"):
	mkdir('./data_textures')
    print('Connecting and downloading Texture Database from Ponce Group...')
    print('1st textures...')
    os.system('wget http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T01-T05.zip')
    print('2nd textures...')
    os.system('wget http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T06-T10.zip')
    print('3rd textures...')
    os.system('wget http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T11-T15.zip')
    print('4th textures...')
    os.system('wget http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T16-T20.zip')
    print('5th textures...')
    os.system('wget http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T21-T25.zip')

    print('Unziping files...')
    os.system('unzip ./T01-T05.zip -d ./data_textures')
    os.system('rm T01-T05.zip ')
    os.system('unzip ./T06-T10.zip -d ./data_textures')
    os.system('rm T06-T10.zip ')
    os.system('unzip ./T11-T15.zip -d ./data_textures')
    os.system('rm T11-T15.zip ')
    os.system('unzip ./T16-T20.zip -d ./data_textures')
    os.system('rm T16-T20.zip ')
    os.system('unzip ./T21-T25.zip -d ./data_textures')
    os.system('rm T21-T25.zip ')

    os.system('rm BSR_bsds500.tgz')
else:
    print('BSDS500 dataset is already downloaded')
