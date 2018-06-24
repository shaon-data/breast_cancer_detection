import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True) #making 99999 is big outlier for missing data. this is ignored or recognized by most of the algorithm as outliers
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1)) #seperating label column
y = np.array(df['class']) #storing only the label column

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2) #20% data for test set, 80% for train set

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([4,2,1,1,1,2,3,2,1])

prediction = clf.predict(example_measures)
print(prediction)
