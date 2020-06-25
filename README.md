# Banknote Forgery Detector (MSA 2020)

## Project Idea

The project aims to use machine learning via a supervised learning model (support vector machine) to classify whether a banknote is authentic or forged. Each banknote has 5 attributes:
1. Variance of Wavelet Transformed image (continuous)
2. Skewness of Wavelet Transformed image (continuous)
3. Curtosis of Wavelet Transformed image (continuous)
4. Entropy of image (continuous)
5. Class i.e. not forged, forged (discrete)

The model itself classifies a banknote's class based on 3 attributes, curtosis, skewness and variance. These 3 attributes were chosen after various visualisations indicated that this triplet showed the most distinction between the two different classes.

The data in the data set was extracted from images of genuine and forged banknotes, using the Wavelet Transform tool. The original data set is from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/banknote+authentication#).

The machine learning model was built using Python, and encapsulated by an object, ForgeryDetector, complete with methods that allow training, testing, data visualisation and predicting the classes of data points.

## Background on Support Vector Machines

Support vector machines (SVMs) were the machine learning model of choice because of their prevalent use in classification problems, a high degree of accuracy and requiring less computational power. SVMs attempt to find an N-dimensional hyperplane, where N is the number of features, that classifies data points (e.g. points on either side of the hyperplane are a part of different classes). The hyperplane is influenced by support vectors (i.e. points close to the hyperplane), attempting to maximise the distance between data points in the different classes.

## Environment Setup & Dependencies

Python 3.6.1 and above are compatible, but Python 3.8.x is recommended and can be installed from [here](https://www.python.org/downloads/).

Required Python libraries (and their pip install commands):
* [Matplotlib](https://matplotlib.org/) (3.2.1 and above)
  * ```pip install matplotlib```
* [NumPy](https://numpy.org/) (1.18.4 and above)
  * ```pip install numpy```
* [pandas](https://pandas.pydata.org/) (1.0.4 and above)
  * ```pip install pandas```
* [scikit-learn](https://scikit-learn.org/stable/) (0.0 and above)
  * ```pip install sklearn```

For more information, refer to their respective documentation.

## Instructions for Using the Model

The machine learning model is simple to use. It is encapsulated by the ForgeryDector object (defined in ```models.py```), and can be initialised by calling ```model = ForgeryDetector('banknote_data.txt')```.

The minimum requirement is the file name of the data set to be used in training and testing the model. A data set is provided (```banknote_data.txt```). 
It is expected that that data set consists of 5 columns (curtosis, entropy, is_forged, skewness, variance) and multiple rows.

When initialising the ForgeryDetector object, by default:
* if the data set does not have headers, it is assumed that it has all 5 columns previously mentioned, in alphabetical order. If it does have headers, pass the argument ```header=0``` into the model constructor e.g. ```model = ForgeryDetector('banknote_data.txt', header=0)```. The data set does not need the columns to be ordered if it has headers as it will be done automatically.
* the data set is split into training and testing data sets, using 50% (proportion of 0.5) for training and 50% (proportion of 0.5) for testing. If desired, this can be changed by passing the argument ```train_split``` into the model constructor with the desired proportion e.g. ```model = ForgeryDetector('banknote_data.txt', train_split=0.8)``` trains the model using 80% (proportion of 0.8) of the data set and tests the model using 20% (proportion of 0.2).

The features of the data set can be visualised in pairs and/or triplets by calling ```model.plot_all()```. By default, the features are visualised in triplets, but pairs of features can be visualised by adding ```plot_pairs``` equal to True to the method call e.g. ```model.plot_all(plot_pairs=True)```. These visualisation were what deemed the use of the featurse curtosis, skewness and variance to train the model (this triplet had the greatest distinction between forged and not forged). 

**Helpful tips:**
* Example usage of the ForgeryDetector object can be found in ```main.py```.
* To zoom in & out of the various plots, right-click, hold and drag to adjust the zoom.

### Training & Testing

The model can be trained & tested simply by initialising the ForgeryDetector object and calling its train_and_test method e.g. ```model.train_and_test()```. 

Using the training data (determined by ```train_split```), the model is trained and then the test data is used to test the model and its accuracy. By default, a 3D scatter plot is displayed, showing the SVM's hyperplane that classifies the data points as forged or not forged and the model's accuracy. A normalised confusion matrix is also displayed, to show which data points were correctly classified and which were not as a percentage. Testing the model can be skipped if desired e.g. ```model.train_and_test(test=False)```.

### Predicting Classes

The purpose of the ForgeryDetector object is to be able to classify whether a banknote was forged or not (to a certain degree of accuracy). After training and testing, further data can be passed into the ```predict``` method of the object (e.g. ```results = model.predict(test_data)```), returning an array of predicted results e.g. ```[0, 1]``` indicates the first banknote (first row) is not forged and the second banknote (second row) is forged.

The test data can be:
* an array of arrays (e.g. ```test_data = [[3.62, 8.66, -2.80], [-2.83, -6.63, 10.48]]```). If the test data is an array of arrays, it is assumed that the columns are  curtosis, skewness and variance, in this order.
* a [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) (e.g. ```test_data = pd.DataFrame([[3.62, 8.66, -2.80, -12], [-2.83, -6.63, 10.48, -8]], columns=['variance', 'curtosis', 'skewness', 'randomcolumn'])```). If the test data is a DataFrame and does not have headers, it is assumed that the columns are curtosis, skewness and variance, in this order e.g. ```test_data = pd.DataFrame([[3.62, 8.66, -2.80], [-2.83, -6.63, 10.48]])```.

## ForgeryDetector Object Methods

The primary methods that can be used:
* ```ForgeryDetector(filename, header=None, train_split=0.5)``` - constructs a new ForgeryDetector object, with the data set's filename, indicator of whether the data set has headers, and the proportion of the data split and used for training
* ```.load_data_set(filename, header)``` - loads a data set and configures it for use, useful to switch to another data set (the model will need to be retrained); done automatically when creating a new instance of the object
* ```.plot_all(plot_pairs=False)``` - plots all triplets of features to show variance, set ```plot_pairs``` to True to plot pairs of features
* ```.train_and_test(test=True)``` - trains the SVM with provided data set and trains it to determine its accuracy, though testing can be skipped by setting ```test``` to False
* ```.predict(test_data)``` - returns array of predicted results, 1 if forged, 0 if not forged based on data provided

It is recommended to use ```.plot_all()``` and avoid using:
* ```.plot_two_features(features)``` - plots any two features of the data set using the features' column names
* ```.plot_three_features(features, plot_all=True, test_data=None)``` - plots any three features of the data set using the features' column names, it can also plot test data, predicted results and accuracy of SVM if ```plot_all``` is False and test data is provided