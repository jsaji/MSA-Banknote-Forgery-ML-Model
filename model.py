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

    def plot_three_features(self, feature_1, feature_2, feature_3):
        fig = graph.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(self.data_set[feature_1], self.data_set[feature_2], self.data_set[feature_3], c=self.data_set['is_forged'])
        print(self.data_set.head())
        ax.set_xlabel(feature_1)
        ax.set_ylabel(feature_2)
        ax.set_zlabel(feature_3)
        title = 'Classification Plot using ' + feature_1 + ', ' + feature_2 + ' & ' + feature_3
        graph.legend(handles=scatter.legend_elements()[0], labels=["not forged", "forged"])
        graph.title(title)
        graph.show()

    def plot_all(self):
        '''Plots every pair of features'''
        '''
        for i in range(len(self.data_set.columns)):
            for j in range(i+1, len(self.data_set.columns), 1):
                self.plot_features(self.data_set.columns[i], self.data_set.columns[j])
        '''
        # Plotting each pair does not provide a conclusive set of features to train with
        '''Plots every triplet of features'''
        total = len(self.data_set.columns) - 1
        for i in range(total):
            for j in range(i + 1, total, 1):
                for k in range(j + 1,total, 1):
                    self.plot_three_features(self.data_set.columns[i], self.data_set.columns[j],self.data_set.columns[k])
        # The most promising combination is variance, curtosis and skewness (this plot had distinctive separation between forged/not forged)
                


    def something(self):
        '''Don't know what to do here yet'''
        all_features = self.data_set.drop(['is_forged'], axis=1)
        variance_feat = self.data_set['variance'].values
        curtosis_feat = self.data_set['curtosis'].values
        skewness_feat = self.data_set['skewness'].values
        entropy_feat = self.data_set['entropy'].values
        is_forged = self.data_set['is_forged'].values

