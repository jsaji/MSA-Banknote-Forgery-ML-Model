'''
model.py
Creates a ML model to analyze data from scanned banknotes to determine if they are legitimate or forged
'''

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as graph
from sklearn import svm

data_set = pd.read_csv('Resources/banknote_data.txt', sep=',', header=None)
data_set.columns = ['variance', 'curtosis', 'skewness', 'entropy', 'is_forged']
data_set = data_set.round(2)

def plotFeatures(feature_1, feature_2, category):
    scatter = graph.scatter(data_set[feature_1], data_set[feature_2], c=data_set[category], marker='D')
    graph.xlabel(feature_1)
    graph.ylabel(feature_2)
    title = 'Classification Plot using ' + feature_1 + ' & ' + feature_2
    graph.title(title)
    graph.legend(handles=scatter.legend_elements()[0], labels=["not forged", "forged"])
    graph.grid()
    graph.show()

for i in range(len(data_set.columns)):
    for j in range(i+1, len(data_set.columns), 1):
        plotFeatures(data_set.columns[i], data_set.columns[j], 'is_forged')

all_features = data_set.drop(['is_forged'], axis=1)
variance_feat = data_set['variance'].values
curtosis_feat = data_set['curtosis'].values
skewness_feat = data_set['skewness'].values
entropy_feat = data_set['entropy'].values
is_forged = data_set['is_forged'].values
