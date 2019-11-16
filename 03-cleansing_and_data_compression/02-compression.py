#!/bin/env python
# 2019/11/16
# data compression example
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# Stanislaw Grams <sjg@fmdx.pl>
# 03-cleansing_and_data_compression/02-compression.py

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

## Import data from file
features = ['sepal length',
        'sepal width',
        'petal length',
        'petal width']

df  = pd.read_csv ("iris.data", names=features + ['target'])
print (df)

## Standardize data
x = df.loc[:, features].values
y = df.loc[:, ['target']].values
x = StandardScaler ().fit_transform (x)

## Apply PCA (n_components = 2)
pca = PCA (n_components = 2)
principalComponents = pca.fit_transform (x)
principalDf = pd.DataFrame (principalComponents, columns = [
    'principal component 0',
    'principal component 1'])

finalDf2d = pd.concat ([principalDf, df[['target']]], axis = 1)

fig = plt.figure (figsize = (8,8))
ax = fig.add_subplot (1,1,1)
ax.set_xlabel ('Principal Component 0', fontsize=14)
ax.set_ylabel ('Principal Component 1', fontsize=14)
ax.set_title  ('2D PCA', fontsize=18)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip (targets,colors):
    indicesToKeep = finalDf2d['target'] == target
    ax.scatter (finalDf2d.loc[indicesToKeep, 'principal component 0'],
            finalDf2d.loc[indicesToKeep, 'principal component 1'],
            c = color,
            s = 50)
ax.legend (targets)
ax.grid ()

print ("Total ratio of two columns")
print (str(pca.explained_variance_ratio_.sum() * 100.0) + "%")
plt.show ()
