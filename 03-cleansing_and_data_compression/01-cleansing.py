#!/bin/env python
# 2019/11/10
# data cleansing example
# Stanislaw Grams <sjg@fmdx.pl>
# 03-cleansing_and_data_compression/01-cleansing.py

import numpy  as np
import pandas as pd

missing_values = ["-", "NA", "--", "n/a", "NaN"]
pd.set_option ('display.max_rows', 500)
df = pd.read_csv('iris_with_errors.csv', na_values = missing_values)

print ("Number of invalid values in corresponding columns")
print (df.isnull().sum())

## replace invalid ranges and n/a in cols with medianas
median_sepal_length = df['sepal.length'].median ()
df['sepal.length'].fillna (median_sepal_length, inplace=True)
df['sepal.length']  = np.where (
        ((df['sepal.length'] > 15) | (df['sepal.length'] <= 0)),
        median_sepal_length,
        df['sepal.length'])

median_sepal_width = df['sepal.width'].median ()
df['sepal.width'].fillna (median_sepal_width, inplace=True)
df['sepal.width'] = np.where (
        ((df['sepal.width'] > 15) | (df['sepal.width'] <= 0)),
        median_sepal_width,
        df['sepal.width'])

median_petal_length = df['petal.length'].median ()
df['petal.length'].fillna (median_petal_length, inplace=True)
df['petal.length']  = np.where (
        ((df['petal.length'] > 15) | (df['petal.length'] <= 0)),
        median_petal_length,
        df['petal.length'])

median_petal_width = df['petal.width'].median ()
df['petal.width'].fillna (median_petal_width, inplace=True)
df['petal.width'] = np.where (
        ((df['petal.width'] > 15) | (df['petal.width'] <= 0)),
        median_petal_width,
        df['petal.width'])

print ("\nNumber of invalid values in corresponding columns after correction")
print (df.isnull().sum())

## fix coherency in variety naming
vt = df['variety']

print ("\nNumber of different values of all varieties")
print (vt.value_counts ())

## replace invalid values with corrected ones
vt = vt.replace ('virginica', 'Virginica')\
        .replace ('setosa', 'Setosa')\
        .replace ('Versicolour', 'Versicolor')

print ("\nNumber of different values of all varieties after correction")
print (vt.value_counts ())
