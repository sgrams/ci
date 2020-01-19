#!/bin/env python
# 2020/01/18
# Convolution Neural Network
# 07-neural_networks/04-cats_and_dogs-classification.py
import sys, os, cv2, regex, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

## make sure random function is random
random.seed (a=None, version=2)

img_w = 150
img_h = 150

ASSETS_DIR = './04-assets_dir/'
TRAIN_DIR  = ASSETS_DIR + 'train/'
TEST_DIR   = ASSETS_DIR + 'test/'

train_set = [TRAIN_DIR+i for i in os.listdir (TRAIN_DIR)]
test_set  = [TEST_DIR+i for i in os.listdir (TEST_DIR)]

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
        x.append (cv2.resize (cv2.imread (image), (img_w, img_h), interpolation=cv2.INTER_CUBIC))
    return x, y

test_set.sort (key=natural_keys)
train_set.sort (key=natural_keys)
train_set = train_set[0:1300] + train_set[12500:13800]

X, Y                           = prepare_data (train_set)
X_train, X_val, Y_train, Y_val = train_test_split (X, Y, test_size=0.2, random_state=1)

nb_train_samples      = len (X_train)
nb_validation_samples = len (X_val)
batch_size = 16

## Sequential Neural Network from Keras package
model = Sequential ()

model.add (Conv2D (32, 3, 3, border_mode='same', input_shape=(img_w, img_h, 3), activation='relu'))
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

train_generator = train_datagen.flow (np.array (X_train), Y_train, batch_size=batch_size)
validation_generator = val_datagen.flow (np.array (X_val), Y_val, batch_size=batch_size)

history = model.fit_generator (
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

X_test, Y_test = prepare_data (test_set)
test_datagen = ImageDataGenerator (rescale=1. / 255)

test_generator = val_datagen.flow(np.array(X_test), batch_size=batch_size)
prediction_probabilities = model.predict_generator(test_generator, verbose=1)

counter = range(1, len (test_set) + 1)
solution = pd.DataFrame ({"id": counter, "label":list(prediction_probabilities)})
cols = ['label']

for col in cols:
    solution[col] = solution[col].map (lambda x: str(x).lstrip('[').rstrip(']')).astype(float)

solution.to_csv (ASSETS_DIR + "/04-cats_and_dogs.csv", index = False)

## plot testing
probabilities = model.predict_generator(test_generator, TEST_SIZE)
for index, probability in enumerate(probabilities):
    image_path = test_data_dir + "/" +test_generator.filenames[index]
    img = mpimg.imread(image_path)
    with open(TEST_FILE,"a") as fh:
        fh.write(str(probability[0]) + " for: " + image_path + "\n")
    plt.imshow(img)
    if probability > 0.5:
        plt.title("%.2f" % (probability[0]*100) + "% dog")
    else:
        plt.title("%.2f" % ((1-probability[0])*100) + "% cat")
    plt.show()
