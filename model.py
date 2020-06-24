"""
model.py
Defines models used in the project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as graph
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

class ForgeryDetector():
    """
    Defines a support vector machine (SVM) to analyze data from scanned banknotes to determine if they're legitimate or forged
    """
    def __init__(self, filename, header, train_split=0.8):
        try:
            data_set = pd.read_csv(filename, header=header)
            data_set.columns = ['variance', 'curtosis', 'skewness', 'entropy', 'is_forged']
            self.data_set = data_set
            self.svm_model = None
            self.train_split = train_split
        except Exception as err:
            raise err

    def plot_two_features(self, feature_1, feature_2):
        """Plots any two features of the data set using their column name"""
        scatter = graph.scatter(self.data_set[feature_1], self.data_set[feature_2], c=self.data_set['is_forged'], marker='D')
        graph.xlabel(feature_1)
        graph.ylabel(feature_2)
        graph.title('Classification Plot using ' + feature_1 + ' & ' + feature_2)
        graph.legend(handles=scatter.legend_elements()[0], labels=['not forged', 'forged'])
        graph.grid()
        graph.show()

    def plot_three_features(self, feature_1, feature_2, feature_3):
        """Plots any three features of the data set using their column name"""
        fig = graph.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(self.data_set[feature_1], self.data_set[feature_2], self.data_set[feature_3], c=self.data_set['is_forged'])
        ax.set_xlabel(feature_1)
        ax.set_ylabel(feature_2)
        ax.set_zlabel(feature_3)
        graph.legend(handles=scatter.legend_elements()[0], labels=['not forged', 'forged'])
        graph.title('Classification Plot using ' + feature_1 + ', ' + feature_2 + ' & ' + feature_3 + ' (all data)')
        graph.show()

    def plot_all(self, plot_pairs=False):
        """Plots all pairs/triplets of features
        to show that variance, curtosis and skewnewss form the best set of features to train the SVM"""

        # Plots all pair combinations of features
        if (plot_pairs):
            for i in range(len(self.data_set.columns)):
                for j in range(i+1, len(self.data_set.columns), 1):
                    self.plot_two_features(self.data_set.columns[i], self.data_set.columns[j])
        # Plotting pairs of features does not provide a conclusive set of features to train the Ml model with

        # Plots all triplet combinations of features
        num_features = len(self.data_set.columns) - 1
        for i in range(num_features):
            for j in range(i + 1, num_features, 1):
                for k in range(j + 1,num_features, 1):
                    self.plot_three_features(self.data_set.columns[i], self.data_set.columns[j],self.data_set.columns[k])
        # The most promising combination is variance, curtosis and skewness
        # as it has distinctive separation between forged/not forged banknotes
        
    def train_and_test(self, display_test=True):
        """Trains the SVM with provided data set and trains it to determine its accuracy"""
        try:
            # Splits data set into training and test sets
            features = self.data_set.drop(['entropy', 'is_forged'], 1)
            result = self.data_set['is_forged']
            train_x, test_x, train_y, test_y = train_test_split(features, result, train_size=self.train_split, random_state=42)
            '''
            print(train_x.describe())
            print(train_y.describe())

            print(test_x.describe())
            print(test_y.describe())
            '''
            # Trains the SVM using a linear kernel
            self.svm_model = svm.SVC(kernel='linear').fit(train_x, train_y)

            # Tests the SVM by comparing test data and the SVM's prediction
            pred_y = self.svm_model.predict(test_x)
            accuracy = metrics.accuracy_score(test_y, pred_y)
            print("SVM Accuracy: {:.2%}".format(accuracy))

            # If desired, the test data can be visualised alongside the SVM hyperplane
            if (display_test):
                fig = graph.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Plots the SVM hyperplane by reconstructing its algebraic form
                z = lambda x, y: (-self.svm_model.intercept_[0]-self.svm_model.coef_[0][0]*x-self.svm_model.coef_[0][1]*y) / self.svm_model.coef_[0][2]
                tmp = [round(min(test_x.min(axis=1))), round(max(test_x.max(axis=1)))]
                x, y = np.meshgrid(tmp, tmp)
                ax.plot_surface(x, y, z(x, y), alpha=0.6)
                
                # Plots the test data
                scatter = ax.scatter(test_x['variance'], test_x['curtosis'], test_x['skewness'], c=test_y)
                ax.set_xlabel('variance')
                ax.set_ylabel('curtosis')
                ax.set_zlabel('skewness')
                graph.legend(handles=scatter.legend_elements()[0], labels=['not forged', 'forged'])
                graph.title('Classification plot using variance, curtosis and skewness (test data)\nAccuracy: {:.2%}'.format(accuracy))
                graph.show()
                
        except Exception as err:
            raise err

    def predict(self, test_data):
        """Returns array of predicted results, 1 if forged, 0 if not forged based on data provided
        test_data (type 2d array or DataFrame) must have 3 columns, variance, curtosis and skewness, in this order, and must have at least 1 row"""
        try:
            if self.svm_model is not None:
                return self.svm_model.predict(test_data)
            else:
                raise Exception("The SVM Model has not been trained")
        except Exception as err:
            raise err
