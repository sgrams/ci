#!/bin/env python
# 2020/01/18
# Diabetes classification, neural network with 3 outputs
# 07-neural_networks/03-diabetes_classification.py
import os, cv2, re, random
import numpy as np
import pandas as pd
from keras import layers, models, optimizers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

random.seed(a=None, version=2)

img_width = 150
img_height = 150

TRAIN_DIR = '04-cats_and_dogs-dir/train/'
TEST_DIR  = '04-cats_and_dogs-dir/test/'

train_set = [TRAIN_DIR+i for i in os.listdir (TRAIN_DIR)]
test_set  = [TEST_DIR+i for i in os.listdir (TEST_DIR)]

def atoi (text):
    return int (text) if text.isdigit () else text

def natural_keys (text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def prepare_data (list_of_images):
    x = [] # images as arrays
    y = [] # labels

    for image in list_of_images:
        x.append (cv2.resize (cv2.imread (image), (img_width, img_height), interpolation=cv2.INTER_CUBIC))

    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
            print(i)
        elif 'cat' in i:
            y.append(0)
            print(i)
        #else:
            #print('neither cat nor dog name present in images')
    return x, y

train_set.sort (key=natural_keys)
train_set = train_set[0:1300] + train_set[12500:13800]

test_set.sort (key=natural_keys)

X, Y = prepare_data (train_set)
print (K.image_data_format ())

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=1)

nb_train_samples      = len (X_train)
nb_validation_samples = len (X_val)
batch_size = 16

## Sequential Neural Network from Keras package
model = models.Sequential ()

model.add (layers.Conv2D (32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add (layers.Activation ('relu'))
model.add (layers.MaxPooling2D (pool_size=(2, 2)))

model.add (layers.Conv2D (32, (3, 3)))
model.add (layers.Activation ('relu'))
model.add (layers.MaxPooling2D (pool_size=(2, 2)))

model.add (layers.Conv2D (64, (3, 3)))
model.add (layers.Activation ('relu'))
model.add (layers.MaxPooling2D (pool_size=(2, 2)))

model.add (layers.Flatten ())
model.add (layers.Dense (64))
model.add (layers.Activation ('relu'))
model.add (layers.Dropout (0.5))
model.add (layers.Dense (1))
model.add (layers.Activation ('sigmoid'))

model.compile (loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
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

model.save_weights ('model_weights.h5')
model.save ('model_keras.h5')

X_test, Y_test = prepare_data (test_set) #Y_test in this case will be []
test_datagen = ImageDataGenerator (rescale=1. / 255)

test_generator = val_datagen.flow(np.array(X_test), batch_size=batch_size)
prediction_probabilities = model.predict_generator(test_generator, verbose=1)

counter = range(1, len(test_set) + 1)
solution = pd.DataFrame({"id": counter, "label":list(prediction_probabilities)})
cols = ['label']

for col in cols:
    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)

solution.to_csv ("04-cats_and_dogs-dir/04-cats_and_dogs.csv", index = False)
