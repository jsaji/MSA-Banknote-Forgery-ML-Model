'''
main.py
Used for creating, running and utilising the ForgeryDetector ML object
'''
import pandas as pd
from model import ForgeryDetector
try:
    model = ForgeryDetector('Resources/banknote_data.txt', header=0, train_split=0.5)
    #model.plot_all()
    model.train_and_test()
    test_data = pd.DataFrame([[8.66, -2.80, 3.62], [-6.63, 10.48, -2.83]], columns=['curtosis', 'skewness', 'variance'])
    #test_data = [[8.66, -2.80, 3.62], [-6.63, 10.48, -2.83]]
    results = model.predict(test_data)
    print(results)
except Exception as err:
    print(err)