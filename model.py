import numpy as np
import math
import pandas as pd

def alt(val):
    return round(val, 2)

data = pd.read_csv('Resources/banknote_data.txt', sep=",", header=None)
data.columns = ["variance", "curtosis", "skewness", "entropy", "is_forged"]
print(data)
data = data.round(2)
print(data)
print(alt(5.665))

