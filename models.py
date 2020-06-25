"""
models.py
Defines models used in the project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from matplotlib import cm
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

class ForgeryDetector():
    """
    Defines a support vector machine (SVM) to analyze data from scanned banknotes to determine if they're legitimate or forged
    """
    def __init__(self, filename, header=None, train_split=0.5):
        try:
            self.svm_model = None
            self.data_set = None
            self.load_data_set(filename, header)
            self.train_split = train_split
            # Sets default window size for plot displays
            plot.rcParams['figure.figsize'] = (11, 6)
        except Exception as err:
            raise err

    def load_data_set(self, filename, header):
        """Loads a data set and configures it for use"""
        data_set = pd.read_csv(filename, header=header, skip_blank_lines=True)
        # If the data set does not have headers, set them, if it does, make them lowercase
        if header is None:
            # For a data set without headers, the columns are assumed to be in order
            data_set.columns = ['curtosis', 'entropy', 'is_forged', 'skewness', 'variance']
        else:
            data_set.columns = map(str.lower, data_set.columns)
        # Alphabetically sorts data set columns by their name
        data_set = data_set.reindex(sorted(data_set.columns), axis=1)
        self.data_set = data_set
        # Resets svm model when loading a new data set
        self.svm_model = None
        print('New data set loaded (SVM requires training)')

    def plot_two_features(self, features):
        """Plots any two features of the data set using the features' column names"""
        scatter = plot.scatter(self.data_set[features[0]], self.data_set[features[1]], c=self.data_set['is_forged'], marker='D')
        plot.xlabel(features[0])
        plot.ylabel(features[1])
        plot.title('Classification Plot')
        plot.legend(handles=scatter.legend_elements()[0], labels=['not forged', 'forged'])
        plot.show()

    def plot_three_features(self, features, plot_all=True, test_data=None):
        """Plots any three features of the data set using the features' column names
        Can also plot test data, predicted results and accuracy of SVM if plot_all is False"""
        fig = plot.figure()
        title = 'Data Visualisations'
        # Plots all data or test data if chosen to do so
        if plot_all:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(self.data_set[features[0]], self.data_set[features[1]], self.data_set[features[2]], c=self.data_set['is_forged'])
            ax.set_title('Classification Plot (All Data)')
        else:
            # Plots the SVM hyperplane by reconstructing its algebraic form in terms of z
            # a*x + b*y + c*z + d = 0 -> z = (-d - a*x - b*y)/c
            z = lambda x, y: (-self.svm_model.intercept_[0]-self.svm_model.coef_[0][0]*x-self.svm_model.coef_[0][1]*y) / self.svm_model.coef_[0][2]
            limits = [round(min(test_data[0].min(axis=1)))-1, round(max(test_data[0].max(axis=1)))+1]
            # Sets the x, y points of the corners of the displayed hyperplane for visualisation
            x, y = np.meshgrid(limits, limits)
            ax = fig.add_subplot(121, projection='3d')
            ax.plot_surface(x, y, z(x, y), alpha=0.6, color='b')
            # Plots the test data
            y_pred = self.predict(test_data[0])
            scatter = ax.scatter(test_data[0][features[0]], test_data[0][features[1]], test_data[0][features[2]], c=y_pred)
            accuracy = metrics.accuracy_score(test_data[1], y_pred)
            ax.set_title('Classification Plot (Test Data)')
            
            # Plots a normalised confusion matrix
            cnf_matrix = metrics.plot_confusion_matrix(self.svm_model, test_data[0], test_data[1], display_labels=['not forged', 'forged'],
                            normalize='true', ax=fig.add_subplot(122), cmap=cm.cmap_d['Blues'])
            cnf_matrix.ax_.set_title('Normalized Confusion Matrix')

            title += ' - SVM Accuracy: {:.2%}'.format(accuracy)
            print('Testing completed (SVM accuracy: {:.2%})'.format(accuracy))
            
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        ax.legend(handles=scatter.legend_elements()[0], labels=['not forged', 'forged'], loc='lower right')
        fig.suptitle(title)
        fig.tight_layout()
        plot.show()

    def plot_all(self, plot_pairs=False):
        """Plots all triplets of features - set plot_pairs to True to plot pairs of features
        to show variance, curtosis & skewnewss form the best set of features to train the SVM"""
        # Plots all pair combinations of features (avoids duplicate combinations)
        if plot_pairs:
            for i in range(len(self.data_set.columns)):
                for j in range(i+1, len(self.data_set.columns), 1):
                    self.plot_two_features([self.data_set.columns[i], self.data_set.columns[j]])
        # Plotting pairs of features does not provide a conclusive set of features to train the Ml model with

        # Plots all triplet combinations of features (avoids duplicate combinations)
        num_features = len(self.data_set.columns) - 1
        for i in range(num_features):
            for j in range(i + 1, num_features, 1):
                for k in range(j + 1, num_features, 1):
                    self.plot_three_features([self.data_set.columns[i], self.data_set.columns[j], self.data_set.columns[k]])
        # The most promising combination is variance, curtosis and skewness
        # as it has distinctive separation between forged/not forged banknotes

    def train_and_test(self, test=True):
        """Trains the SVM with provided data set and trains it to determine its accuracy"""
        try:
            # Splits data set into training and test sets
            features = self.data_set.drop(['entropy', 'is_forged'], 1)
            result = self.data_set['is_forged']
            x_train, x_test, y_train, y_test = train_test_split(features, result, train_size=self.train_split, random_state=42)
            '''
            print(x_train.describe())
            print(y_train.describe())
            print(x_test.describe())
            print(y_test.describe())
            '''
            # Trains the SVM using a linear kernel
            self.svm_model = svm.SVC(kernel='linear').fit(x_train, y_train)
            print('Training completed')
            # If desired, the SVM can be tested and the results can be visualised
            if test:
                self.plot_three_features(['curtosis', 'skewness', 'variance'], plot_all=False, test_data=[x_test, y_test])
                
        except Exception as err:
            raise err

    def predict(self, test_data):
        """Returns array of predicted results, 1 if forged, 0 if not forged based on data provided"""
        try:
            if self.svm_model is not None:
                # If test data is a DataFrame and has headers (of string type), order the columns and drop unnecessary ones
                # if test_data is not a DataFrame, it is assumed that the columns are ordered as required
                if (isinstance(test_data, pd.DataFrame) and isinstance(test_data.columns[0], str)):
                    test_data.columns = map(str.lower, test_data.columns)
                    test_data = test_data.reindex(sorted(test_data.columns), axis=1)
                    test_data = test_data[['curtosis', 'skewness', 'variance']]
                results = self.svm_model.predict(test_data)
                print('Data classified')
                return results
            else:
                raise Exception('The SVM Model has not been trained')
        except Exception as err:
            raise err
