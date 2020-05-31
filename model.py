'''
model.py
Defines models used in the project
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as graph
from sklearn import svm


class ForgeryDetector():
    '''
    Defines an ML model to analyze data from scanned banknotes to determine if they're legitimate or forged
    '''
    def __init__(self):
        data_set = pd.read_csv('Resources/banknote_data.txt', sep=',', header=None)
        data_set.columns = ['variance', 'curtosis', 'skewness', 'entropy', 'is_forged']
        data_set = data_set.round(2)
        self.data_set = data_set

    def plot_features(self, feature_1, feature_2):
        '''Plots any two features of the data set using their column name'''
        scatter = graph.scatter(self.data_set[feature_1], self.data_set[feature_2], c=self.data_set['is_forged'], marker='D')
        graph.xlabel(feature_1)
        graph.ylabel(feature_2)
        title = 'Classification Plot using ' + feature_1 + ' & ' + feature_2
        graph.title(title)
        graph.legend(handles=scatter.legend_elements()[0], labels=["not forged", "forged"])
        graph.grid()
        graph.show()

    def plot_all(self):
        '''Plots every pair of features'''
        for i in range(len(self.data_set.columns)):
            for j in range(i+1, len(self.data_set.columns), 1):
                self.plot_features(self.data_set.columns[i], self.data_set.columns[j])

    def something(self):
        '''Don't know what to do here yet'''
        all_features = self.data_set.drop(['is_forged'], axis=1)
        variance_feat = self.data_set['variance'].values
        curtosis_feat = self.data_set['curtosis'].values
        skewness_feat = self.data_set['skewness'].values
        entropy_feat = self.data_set['entropy'].values
        is_forged = self.data_set['is_forged'].values

