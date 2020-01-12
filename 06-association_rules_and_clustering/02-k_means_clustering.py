#!/bin/env python
# 2020/01/11
# K-means clustering
# Stanislaw Grams <sjg@fmdx.pl>
# 06-association_rules_and_clustering/02-k_means_clustering.py
import numpy    as np
import pandas   as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

## data load
dataframe = pd.read_csv ("iris2D.csv", usecols=['PC1', 'PC2'])
x = dataframe.iloc[:, [0, 1]].values

## apply K-means clustering
kmeans = KMeans (n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict (x);

## scatter chart
plt.scatter (x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter (x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='green', label='Cluster 2')
plt.scatter (x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='blue', label='Cluster 3')
plt.scatter (kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s=100, c='magenta', label='Centroids')
plt.legend ()
plt.show ()
