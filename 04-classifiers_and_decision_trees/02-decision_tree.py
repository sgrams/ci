#!/bin/env python
# 2019/11/29
# decision tree example
# Stanislaw Grams <sjg@fmdx.pl>
# 04-classifiers_and_decision_trees/02-decision_tree.py
import pandas   as pd
import graphviz as gv
from sklearn             import tree
from sklearn.metrics     import confusion_matrix
from sklearn.tree        import DecisionTreeClassifier

dataframe = pd.read_csv ("iris.csv")
features  = list (dataframe.drop ('class', axis=1).columns)
classes   = [x for x in list (dataframe['class'].unique ())]

# split dataframe (30/70)
training_set = dataframe.sample (frac=0.70)
testing_set  = dataframe[~dataframe.isin(training_set).all(1)]
print (training_set)
print (testing_set)
print ()

# tree-training on the learning set
decision_tree = tree.DecisionTreeClassifier ()
decision_tree.fit (training_set.drop ('class', axis=1), training_set['class'])

# generate graphviz output
dot_data = tree.export_graphviz (
        decision_tree,
        out_file=None,
        feature_names=features,
        class_names=classes,
        filled=True,
        proportion=True)
graph = gv.Source (dot_data)
graph.render (
        "decision_tree",
        cleanup=True,
        view=False)

# text output of decision tree
print ("Decision tree in ASCII:")
print (tree.export_text (decision_tree, feature_names=features))

# percent of correct solutions
print ("Percent of correct solutions: {}%".format (decision_tree.score (testing_set.drop('class', axis=1), testing_set['class'])))

# confusion matrix
print ("Confusion matrix for n={}".format (len (testing_set.index)))
print (confusion_matrix(testing_set['class'], decision_tree.predict (testing_set.drop ('class', axis=1))))
