import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')
import pandas as pd
import random

benign_class = 2
malignant_class = 4

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
    
    result_label_votes = Counter(votes).most_common(1)[0][1]
    total_votes = k
    confidence = result_label_votes/total_votes
        
    return votes_result,confidence


dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]
'''
trials = 1
accuracies = []
for i in range(trials):

    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace=True)

    #feature_data = df.drop(['class'],1).astype(float).values.tolist()
    #target_data = df['class'].astype(float).values.tolist()
    full_data = df.astype(float).values.tolist()

    #Shuffling before creating test train set from keeping it out of any bias
    random.shuffle(full_data)

    ## creating dictionary with unique value of target column 2,4 - benign class, malignant class
    train_set = {}
    test_set = {}
    for dicn in df['class'].unique():
        train_set[dicn] = []
        test_set[dicn] = []
    # creating dictionary with unique value of target column

    ##train_set = {benign_class:[],malignant_class:[]} 
    ##test_set = {benign_class:[], malignant_class:[]} # creating dictionary of two classes

    #creating test train set
    test_size = 0.5
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


    ses_con = []
    for x in range(17):
        k = 2*x+1
        print("k=%s"%k)

        correct = 0
        total = 0
        confidences = []
        for train_label in test_set:
            for row in test_set[train_label]:
                result,confidence = k_nearest_neighbors(train_set,row,k)
                confidences.append(confidence)

                #if confidence < 0.6:
                #    print(result,confidence)

                if train_label == result:
                    correct += 1
                else:
                    print(confidence)
                total += 1
        print("Accuracy",correct/total)
        avg_confidence = sum(confidences)/total
        print("avg confidence = %s"%avg_confidence)
        accuracies.append(correct/total)

        ses_con.append([correct/total,k])

    votes = [i for i in sorted(ses_con,reverse=True)[:]] #top k numbers distances
    print(test_size)
    print(votes)

        

model_accuracy = sum(accuracies)/len(accuracies)
print("Session Model range %s - %s"%(min(accuracies),max(accuracies)))
print("Session Model accuracy %s"%(model_accuracy))
# split = 20%,40% k = 7, accuracy = 1, 96%
'''
# prediction
df = pd.read_csv('samples.txt')
#df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
predict_data = df.astype(float).values.tolist()
#k_nearest_neighbors(data,predict,k=3)


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

## creating dictionary with unique value of target column 2,4 - benign class, malignant class
full_data_set = {}    
for dicn in df['class'].unique():
    full_data_set[dicn] = []
## creating dictionary with unique value of target column 2,4 - benign class, malignant class

for row in full_data:
    # when last element of the row is target label
    label = row[-1] # last element
    row_without_label = row[:-1] # all element untill without and untill last element
    full_data_set[label].append(row_without_label)

ses_con = []
for x in range(4):
    k=2*x + 1
    
    print("k=%s"%k)
    print("Total samples = %s"%(len(predict_data)))
    Cconfidences = []
    for row in predict_data:
        
        result,confidence = k_nearest_neighbors(full_data_set,row,k)
        if result == benign_class:
            con="benign"
        elif result == malignant_class:
            con="malignant"
            
        print("%s , %s" %(con,confidence))
        Cconfidences.append(confidence)

    pred_accur = sum(Cconfidences)/len(Cconfidences)
    print("PRED ACCur = %s"%(pred_accur)) 
    ses_con.append([pred_accur, k])

votes = [i[1] for i in sorted(ses_con,reverse=True)[:]] #top k numbers distances
print(votes)
#votes_result = ( Counter(votes).most_common(1) )[0][0] #most_common(n) most appeared n numbers element among given top least distant k votes , [0][0] first of first element inside list inside tuple    
# prediction


##[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
##plt.scatter(new_features[0],new_features[1], color = result)
##plt.title('KNN')
##plt.show()
