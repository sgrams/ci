#!/bin/env python
# 2019/11/29
# classifier example
# Stanislaw Grams <sjg@fmdx.pl>
# 04-classifiers_and_decision_trees/01-classifier.py
import numpy  as np
import pandas as pd

def myPredictRow (df):
    if (df["petalwidth"] <= 0.6):
        return 'Iris-setosa'
    else:
        if (df["petalwidth"] <= 1.6):
            return 'Iris-versicolor'
        else:
            return 'Iris-virginica'

def myCompare (df):
    return myPredictRow (df) == df["class"]

dataframe = pd.read_csv ("iris.csv")
print ("Accuracy of myPredictRow function: " +\
        str(dataframe.apply (myCompare, axis=1).mean () * 100) + "%")
