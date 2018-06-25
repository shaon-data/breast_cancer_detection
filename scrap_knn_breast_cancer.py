import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')
import pandas as pd
import random

def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = sqrt( (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 ) for 2D
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]] #top k numbers distances
    votes_result = ( Counter(votes).most_common(1) )[0][0] #most_common(n) most appeared n numbers element among given top least distant k votes , [0][0] first of first element inside list inside tuple
##    print(votes)
##    print(votes_result)
    
    return votes_result
    

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

feature_data = df.drop(['class'],1).astype(float).values.tolist()
target_data = df['class'].astype(float).values.tolist()
full_data = df.astype(float).values.tolist()

#creating test train set
random.shuffle(full_data) #shuffling

test_size = 0.2

train_set = test_set = {}
for dicn in df['class'].unique():
    train_set[dicn] = test_set[dicn] = []
#train_set = {2:[],4:[]} # creating dictionary of two classes
#test_set = {2:[], 4:[]} # creating dictionary of two classes
train_data = full_data[:-int(test_size*len(full_data))] #first 80% of the data
test_data = full_data[-int(test_size*len(full_data)):] #first 20% of the data




result = k_nearest_neighbors(dataset,new_features,k=3)

[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1], color = result)
plt.title('KNN')
plt.show()
