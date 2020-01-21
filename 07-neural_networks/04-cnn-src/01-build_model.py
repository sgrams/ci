#!/bin/env python
# 2020/01/18
# Convolution Neural Network
# 07-neural_networks/04-cnn-src/01-build_model.py
from CatsDogsCommon import *
from datetime import datetime

from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.callbacks import CSVLogger
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

## prepare data for training
X, Y                           = prepare_data (train_set)
## split training data 80%/20% into training and validation subsets
X_train, X_val, Y_train, Y_val = train_test_split (X, Y, test_size=0.2, random_state=1)

nb_train_samples      = len (X_train) ## number of training samples
nb_validation_samples = len (X_val)   ## number of validation samples
batch_size = 16

## Sequential Neural Network model from Keras package
## see https://towardsdatascience.com/image-classifier-cats-vs-dogs-with-convolutional-neural-networks-cnns-and-google-colabs-4e9af21ae7a8
## for model 5
model = Sequential ()

model.add (Conv2D (32, 3, 3, border_mode='same', input_shape=(IMG_W, IMG_H, 3), activation='relu'))
model.add (Conv2D (32, 3, 3, border_mode='same', activation='relu'))
model.add (MaxPooling2D (pool_size=(2, 2)))

model.add (Conv2D (64, 3, 3, border_mode='same', activation='relu'))
model.add (Conv2D (64, 3, 3, border_mode='same', activation='relu'))
model.add (MaxPooling2D (pool_size=(2, 2)))

model.add (Conv2D (128, 3, 3, border_mode='same', activation='relu'))
model.add (Conv2D (128, 3, 3, border_mode='same', activation='relu'))
model.add (MaxPooling2D (pool_size=(2, 2)))

model.add (Conv2D (256, 3, 3, border_mode='same', activation='relu'))
model.add (Conv2D (256, 3, 3, border_mode='same', activation='relu'))
model.add (MaxPooling2D (pool_size=(2, 2)))

model.add (Flatten ())
model.add (Dense (256, activation='relu'))
model.add (Dropout (0.5))

model.add (Dense (256, activation='relu'))
model.add (Dropout (0.5))

model.add (Dense (1))
model.add (Activation ('sigmoid'))

model.compile (loss='binary_crossentropy',
            optimizer=RMSprop (lr=0.0001),
            metrics=['accuracy'])

model.summary ()

train_datagen = ImageDataGenerator (
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
            )

val_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
            )

train_generator = train_datagen.flow (np.array (X_train), Y_train, batch_size=BATCH_SIZE)
validation_generator = val_datagen.flow (np.array (X_val), Y_val, batch_size=BATCH_SIZE)

history = model.fit_generator (
    train_generator,
    steps_per_epoch=nb_train_samples // BATCH_SIZE,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // BATCH_SIZE
)

cur_datetime = datetime.now ().strftime ("%d%m%Y-%H%M%S")
model.save (ASSETS_DIR + '/cnn-MODEL-' + cur_datetime + '.keras_model');

import json
with open(ASSETS_DIR + '/cnn-HISTORY-' + cur_datetime + '.json', 'w') as f:
        json.dump (history.history, f)
