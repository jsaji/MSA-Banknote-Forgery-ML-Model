# MSA Forgery ML Model

## Project Idea

This project is aimed at using machine learning via classification models to determine whether a banknote is authentic or forged. The model itself classifies a banknote as genuine or forged (discrete) based on 4 attributes:
1. Variance of Wavelet Transformed image (continuous)
2. Skewness of Wavelet Transformed image (continuous)
3. Curtosis of Wavelet Transformed image (continuous)
4. Entropy of image (continuous)

The data in the data set was extracted from images of genuine and forged banknotes, using the Wavelet Transform tool. The original data set is from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/banknote+authentication#).

## Environment Setup & Dependencies

Any Python version 3.0 and above is compatible, but Python 3.8.x is recommended. Python can be installed from [here](https://www.python.org/downloads/).

Required Python libraries:
* [Matplotlib](https://matplotlib.org/) (3.2.1)
* [NumPy](https://numpy.org/) (1.18.4)
* [pandas](https://pandas.pydata.org/) (1.0.4)
* [scikit-learn](https://scikit-learn.org/stable/) (0.0)

These can be installed using pip. For more information, refer to  documentation.

## Step-by-step Instructions for Using the Model

The machine learning model is simple to use. It is encapsulated by the ForgeryDector object (defined in ```models.py```), and can be initialised calling ```model = ForgeryDetector('Resources/banknote_data.txt')```.

The minimum requirement is the file name of the data set to be used in training and testing the model. One data set is provided (```Resources/banknote_data.txt```). It is expected that that data set consists of 5 columns (variance, curtosis, skewness, entropy, is_forged) and multiple rows.

When initialising the ForgeryDetector object, by default:
* it is assumed that the data does not have headers. If it does, pass the argument 'header' into the model constructor to indicate that headers are in the first line of the data file e.g. ```model = ForgeryDetector('Resources/banknote_data.txt', header=0)```.
* the data set is split into training and testing data sets, using 60% (proportion of 0.6) for training and 40% (proportion of 0.4) for testing. If desired, this can be changed by passing the argument 'train_split' into the model constructor with the desired proportion e.g. ```model = ForgeryDetector('Resources/banknote_data.txt', train_split=0.8)``` trains the model using 80% (proportion of 0.8) of the data set.

The features of the data set can be visualised in pairs and/or triplets by calling ```model.plot_all()```. By default, the features are visualised in triplets, but pairs of features can be visualised by calling ```model.plot_all(plot_pairs=True)```.

### Training & Testing

The model can be trained & tested simply by initialising the ForgeryDetector object and calling its train_and_test method e.g. ```model.train_and_test()```. 

Using the training data (determined by train_split), the model is trained and its accuracy is measured (and printed to the console).

By default, it also displays a 3D scatter plot of the support vector machine hyperplane that classifies the data points as forged or not forged and the model's accuracy, but can be disabled if desired e.g. ```model.train_and_test(display_test=False)```.

### Predicting Values

The purpose of the ForgeryDetector object is to be able to predict/determine whether a banknote was forged or not (to a certain degree of accuracy). After training and testing, further data can be passed into the 'predict' method of the object (e.g. ```results = model.predict(test_data)```), returning an array of predicted results e.g. ```[0, 1]``` indicates the first banknote (first row) is not forged and the second banknote (second row) is forged. The test data can be an array of arrays (e.g. ```test_data = [[3.62, 8.66, -2.80], [-2.83, -6.63, 10.48]]```) or a [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) (e.g. ```test_data = pd.DataFrame([[3.62, 8.66, -2.80], [-2.83, -6.63, 10.48]], columns=['variance', 'curtosis', 'skewness'])```). However, the test data is required to have 3 columns, variance, curtosis and skewness, in this order, to use the model (column headers are not required).