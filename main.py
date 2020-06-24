'''
main.py
Used for creating, running and utilising the ForgeryDetector ML object
'''
import pandas as pd
from model import ForgeryDetector

model = ForgeryDetector('Resources/banknote_data.txt', 0.8)
#model.plot_all()
model.train_and_test()
test_data = pd.DataFrame([[3.6216,8.6661,-2.8073],[-2.8391,-6.63,10.4849]], columns=['variance', 'curtosis', 'skewness'])
pred = (model.predict(test_data))
print(str(pred))
