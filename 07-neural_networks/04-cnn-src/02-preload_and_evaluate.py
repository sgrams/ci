#!/bin/env python
# 2020/01/21
# Convolution Neural Network
# 07-neural_networks/04-cnn-src/02-preload_and_evaluate.py
import sys
import getopt
argv = sys.argv

if (len (argv) < 2):
    print (str (argv[0]) + ": exited, please provide a valid keras model")
    sys.exit (1)

## perform evaluation on model
from CatsDogsCommon import *
from matplotlib import pyplot as plt
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.callbacks import CSVLogger
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from datetime import datetime

model = load_model (argv[1])
model.summary ()

X_test, Y_test = prepare_data (test_set)
test_datagen = ImageDataGenerator (rescale=1. / 255)
test_generator = test_datagen.flow(np.array(X_test), batch_size=BATCH_SIZE)
probabilities = model.predict_generator (test_generator, verbose=1)

## perform testing and display 25 randomly chosen samples
import random
fig = plt.figure ()
img_num = 0
for index, probability in random.sample (list (enumerate (probabilities)), k=25):
    img_num += 1
    y = fig.add_subplot (5, 5, img_num, ymargin=15, xmargin=15)
    y.imshow (X_test[index])
    y.set_aspect('equal')
    plt.tight_layout ()
    if probability > 0.5:
        plt.title ("%.2f" % (probability[0] * 100) + "% dog")
    else:
        plt.title ("%.2f" % ((1-probability[0])*100) + "% cat")
    y.axes.get_xaxis ().set_visible (False)
    y.axes.get_yaxis ().set_visible (False)
cur_datetime = datetime.now ().strftime ("%d%m%Y-%H%M%S")
fig.savefig (ASSETS_DIR + '/cnn-SAMPLES-' + cur_datetime + '.png', format='png')

## perform evaluation
loss, acc = model.evaluate (np.array(X_test), Y_test, batch_size=BATCH_SIZE, verbose=1)
print ("overall evaluation loss = " + str (loss))
print ("overall evaluation accuracy =" + str (acc))
