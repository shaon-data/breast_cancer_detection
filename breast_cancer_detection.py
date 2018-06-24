import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True) #making 99999 is big outlier for missing data. this is ignored or recognized by most of the algorithm as outliers
df.drop(['id'], 1, inplace=True)
