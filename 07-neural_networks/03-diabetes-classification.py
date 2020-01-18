#!/bin/env python
# 2020/01/18
# Diabetes classification, neural network with 3 outputs
# 07-neural_networks/03-diabetes_classification.py
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

    def fit (self, x, y, epochs=200, verbose=2, batch_size=10):
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
    if name == 'tested_positive':
        return 1, 0
    if name == 'tested_negative':
        return 0, 1
    raise ValueError

## load data from file
factors = ['pregnant-times',
        'glucose-concentr',
        'blood-pressure',
        'insulin',
        'mass-index',
        'pedigree-func',
        'age'
        ]
results = ['positive', 'negative']
dataframe = pd.read_csv ("diabetes.csv")

## normalize data
dataframe[factors] = dataframe[factors].apply (lambda x: (x - x.min ()) / (x.max () - x.min ()))

## split data per results
positive = []
negative = []
for index, row in dataframe.iterrows ():
    result = map_result (row['class'])
    positive.append (result[0])
    negative.append (result[1])

dataframe['positive'] = positive
dataframe['negative'] = negative

training_set = dataframe.sample (frac=0.7)
testing_set  = dataframe[~dataframe.isin(training_set).all(1)]

## set-up neural network
net = NeuralNetwork () \
            .add_layer (Dense(10, input_shape=(7,), activation='relu', name='fc1')) \
            .add_layer (Dense(10, activation='relu', name='fc2')) \
            .add_layer (Dense(2, activation='softmax', name='output')) \
            .compile ()

net.fit (training_set[factors], training_set[results])
net.evaluate (testing_set[factors], testing_set[results])

result_set_pred  = net.predict (testing_set[factors])
result_set_pred = (result_set_pred > 0.5).astype(int)
conf_matrix      = multilabel_confusion_matrix (testing_set[results], result_set_pred)

print ("confusion matrix:")
print (conf_matrix)

print (f"accuracy = {net.evaluation_result[1]}")
print (f"loss     = {net.evaluation_result[0]}")
