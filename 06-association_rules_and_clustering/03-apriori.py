#!/bin/env python
# 2020/01/11
# Apriori algorithm
# Stanislaw Grams <sjg@fmdx.pl>
# 06-association_rules_and_clustering/03-apriori.py
import csv
import numpy    as np
import pandas   as pd
import matplotlib.pyplot as plt
from apyori import apriori
from mlxtend.frequent_patterns import association_rules

class Rule:
    def __init__ (self, base, conclusions, support, confidence, lift):
        self.base = base
        self.conclusions = conclusions
        self.support = support
        self.confidence = confidence
        self.lift = lift
    def __repr__ (self):
        return str (self.base) + " => " + str (self.conclusions) + "\n" \
                + "support=" + str (self.support) + ", " \
                + "lift=" + str (self.lift) + ", " \
                + "confidence=" + str (self.confidence)
#    def isAboutSex (self):
#        return 'Male' in self.conclusions or 'Female' in self.conclusions
#    def isAboutAge (self):
#        return 'Child' in self.conclusions or 'Adult' in self.conclusions
    def isAboutSurvival (self):
        return 'Yes' in self.conclusions or 'No' in self.conclusions

## data load & cleansing
rm_quote = lambda x: x.replace ('"', '')
dataframe = pd.read_csv ("titanic.csv", quoting=csv.QUOTE_NONE, engine='python').replace('"', '', regex=True)
dataframe = dataframe.rename (columns=rm_quote)
dataframe = dataframe.drop (dataframe.columns[0], axis='columns')
print (dataframe)

## apriori
result = apriori (dataframe.values, min_support=0.005, min_confidence=0.8, use_colnames=True)
result_list = list ()

## association rules
for item in result:
    result_list.append (Rule (
        list (item.ordered_statistics[0].items_base),
        list (item.ordered_statistics[0].items_add),
        item.support,
        item.ordered_statistics[0].confidence,
        item.ordered_statistics[0].lift
        ))
result_list.sort (key=lambda x: x.confidence, reverse=True)
result_survival_list = list ()
result_survival_confidence = list ()
rules = list ()

for item in result_list:
    if item.isAboutSurvival ():
        result_survival_list.append (item)

## print association rules
print ("\n\nRules about survival:")
for item in result_survival_list:
    tmp_string = str (item.base) + " => " + str (item.conclusions)
    rules.append (tmp_string)
    result_survival_confidence.append (item.confidence)
    print (item)

print ("\n\nAll rules (confidence ≥ 80%):")
for item in result_list:
    print (item)

## bar chart
#plt.bar (rules, result_survival_confidence, align='center')
#plt.ylabel ("confidence")
#plt.xlabel ("rule")
#plt.legend ()
#plt.show ()
