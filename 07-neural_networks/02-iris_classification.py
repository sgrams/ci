#!/bin/env python
# 2020/01/12
# Iris classification, neural network with 3 outputs
# Stanislaw Grams <sjg@fmdx.pl>
# 07-neural_networks/02-iris_classification.py
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix

###
### NeuralNetwork class BEGIN
###
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

class NeuralNetwork ():
    def __init__ (self):
        self.model             = Sequential ()
        self.train_history     = None
        self.evaluation_result = None

    def add_layer (self, layer):
        self.model.add (layer)
        return self

    def compile (self, optimizer=Adam(lr=0.001), loss='categorical_crossentropy'):
        self.model.compile (optimizer, loss=loss, metrics=['accuracy'])
        return self

    def fit (self, x, y, epochs=500, verbose=2, batch_size=10):
        self.train_history = self.model.fit (x, y, verbose=verbose, batch_size=batch_size, epochs=epochs)
        return self

    def evaluate (self, x, y):
        self.evaluation_result = self.model.evaluate (x, y)
        return self

    def predict (self, x):
        return self.model.predict (x)
###
### NeuralNetwork class END
###

def map_result (name):
    if name == 'Iris-setosa':
        return 1, 0, 0
    if name == 'Iris-versicolor':
        return 0, 1, 0
    if name == 'Iris-virginica':
        return 0, 0, 1
    raise ValueError

## load data from file
features = ['sepallength',
        'sepalwidth',
        'petallength',
        'petalwidth']
species = ['setosa', 'versicolor', 'virginica']

dataframe = pd.read_csv ("iris.csv")

## normalize data
dataframe[features] = dataframe[features].apply (lambda x: (x - x.min ()) / (x.max () - x.min ()))

## split data per species
setosa     = []
versicolor = []
virginica  = []
for index, row in dataframe.iterrows ():
    result = map_result (row[4])
    setosa.append (result[0])
    versicolor.append (result[1])
    virginica.append (result[2])

dataframe['setosa']     = setosa
dataframe['versicolor'] = versicolor
dataframe['virginica']  = virginica

training_set = dataframe.sample (frac=0.7)
testing_set  = dataframe[~dataframe.isin(training_set).all(1)]

## set-up neural network
net = NeuralNetwork () \
            .add_layer (Dense(10, input_shape=(4,), activation='relu', name='fc1')) \
            .add_layer (Dense(10, activation='relu', name='fc2')) \
            .add_layer (Dense(3, activation='softmax', name='output')) \
            .compile ()

net.fit (training_set[features], training_set[species])
net.evaluate (testing_set[features], testing_set[species])

result_set_pred  = net.predict (testing_set[features])
result_set_pred = (result_set_pred > 0.5).astype(int)
conf_matrix      = multilabel_confusion_matrix (testing_set[species], result_set_pred)

print ("confusion matrix:")
print (conf_matrix)

print (f"accuracy = {net.evaluation_result[1]}")
print (f"loss     = {net.evaluation_result[0]}")
