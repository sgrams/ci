#!/bin/env python
# 2019/12/13
# kNN and Naive Bayes example
# Stanislaw Grams <sjg@fmdx.pl>
# 05-classifiers-knn_and_naive_bayes/03-knn_and_naive_bayes.py
import numpy    as np
import pandas   as pd
import matplotlib.pyplot as plt
from sklearn             import tree
from sklearn.metrics     import confusion_matrix
from sklearn.tree        import DecisionTreeClassifier
from sklearn.neighbors   import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def knn (training_set, k):
    knn = KNeighborsClassifier (n_neighbors=k)
    knn.fit (training_set.drop ('class', axis=1), training_set['class'])
    return knn

def gnb (training_set):
    gnb = GaussianNB ()
    gnb.fit (training_set.drop ('class', axis=1), training_set['class'])
    return gnb

def tree (training_set):
    tree = DecisionTreeClassifier ()
    tree.fit (training_set.drop ('class', axis=1), training_set['class'])
    return tree

def get_score (classifier, testing_set):
    return classifier.score (testing_set.drop ('class', axis=1), testing_set['class'])

def get_confusion_matrix (classifier, testing_set):
    return confusion_matrix(testing_set['class'], classifier.predict (testing_set.drop ('class', axis=1)))

dataframe = pd.read_csv ("diabetes.csv")
classes   = [x for x in list (dataframe['class'].unique ())]

# split dataframe (67% / 33%)
training_set = dataframe.sample (frac=0.67)
testing_set  = dataframe[~dataframe.isin(training_set).all(1)]
print (training_set)
print (testing_set)

nn3  = knn (training_set, 3)
nn5  = knn (training_set, 5)
nn11 = knn (training_set, 11)
tree = tree (training_set)
gnb  = gnb (training_set)


functions   = [nn3, nn5, nn11, tree, gnb]
classifiers = ["3NN", "5NN", "11NN", "tree", "naive_bayes"]
scores      = [get_score (x, testing_set) for x in functions]

for classifier, function, score in zip (classifiers, functions, scores):
    print ("\nconfusion matrix for " + classifier + " classifier")
    print (get_confusion_matrix (function, testing_set))
    print ("accuracy = " + str (score))

## bar chart
plt.bar (classifiers, scores, align='center')
plt.ylabel ("Scores")
plt.xlabel ("Classifier")
plt.show ()
