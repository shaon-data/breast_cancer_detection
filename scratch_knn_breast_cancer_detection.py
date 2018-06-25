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

# creating dictionary with unique value of target column
train_set = test_set = {}
for dicn in df['class'].unique():
    train_set[dicn] = test_set[dicn] = []
# creating dictionary with unique value of target column

#train_set = {2:[],4:[]} 
#test_set = {2:[], 4:[]} # creating dictionary of two classes

#Shuffling before creating test train set from keeping it out of any bias
random.shuffle(full_data)

#creating test train set
test_size = 0.2
train_data = full_data[:-int(test_size*len(full_data))] #first 80% of the data
test_data = full_data[-int(test_size*len(full_data)):] #first 20% of the data
#creating test train set

# storing test and train data in target key dictionary
for row in train_data:
    # when last element of the row is target label
    label = row[-1] # last element
    row_without_label = row[:-1] # all element untill without and untill last element
    train_set[label].append(row_without_label)

for row in test_data:
    # when last element of the row is target name or label
    label = row[-1] # last element
    row_without_label = row[:-1] # all element untill without and untill last element
    test_set[label].append(row_without_label)
# storing test and train data in target key dictionary

correct = 0
total = 0

for train_label in test_set:
    for row in test_set[train_label]:
        vote = k_nearest_neighbors(train_set,row,k=5)
        if train_label == vote:
            correct += 1
        total += 1
print('Accuracy %s'%(correct/total))

##[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
##plt.scatter(new_features[0],new_features[1], color = result)
##plt.title('KNN')
##plt.show()
