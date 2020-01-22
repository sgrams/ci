#!/bin/env python
# 2020/01/21
# Convolution Neural Network
# Stanislaw Grams <sjg@fmdx.pl>
# 07-neural_networks/04-cnn-src/CatsDogsCommon.py
import os, cv2, regex, random

ASSETS_DIR = '../04-cnn-assets/'
IMG_DIR    = ASSETS_DIR + 'img/'
BATCH_SIZE = 16

IMG_W = 150
IMG_H = 150

def atoi (text):
    return int (text) if text.isdigit () else text

def natural_keys (text):
    ## WARNING, DANGEROUS FUNCTION
    ## WILL RUIN YOUR LIFE, make sure you combine regex with
    ## paths that don't include names used in prepare_data function
    return [ atoi(c) for c in regex.split("(\d+)", text) ]

def prepare_data (list_of_images):
    x = [] # images as arrays
    y = [] # labels

    for i in list_of_images:
        if 'cat' in i:
            y.append (0)
        elif 'dog':
            y.append (1)
    for image in list_of_images:
        x.append (cv2.resize (cv2.imread (image), (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC))
    return x, y

## make sure random function is random
random.seed (a=None, version=2)

## create list of all images
all_images = [IMG_DIR+i for i in os.listdir (IMG_DIR)]
all_images.sort (key=natural_keys)

## split all images into train and test sets
train_set = all_images[0:10000] + all_images[12500:22500]     ## train on 10k images of each class (20k images)
test_set  = all_images[10000:12500] + all_images[22500:25000] ## test on 2.5k images of each class (5k images)
