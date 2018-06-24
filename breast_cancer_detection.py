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
print("Accuracy of the model = %s"%accuracy)

#predicting_sample_inputs
print("Taking Input from Samples.txt ....")
sdf = pd.read_csv('samples.txt')

ids = sdf['id'].as_matrix(columns=None)
samples = sdf.as_matrix( columns=['clump_thickness', 'uniformity_of_cell_size','uniformity_of_cell_shape', 'marginal_adhesion','single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses'] ) #converting_to_2_dimensional_array

example_measures = np.array(samples)
example_measures = example_measures.reshape(len(example_measures),-1) #removing_error

prediction = clf.predict(example_measures)

i = 0
for p,s,idd in zip(prediction,samples,ids):
    i+=1
    if p == 2:
        con = "benign"
    elif p == 4:
        con = "malignant"

    print("Sample #%d -Sl. No%s- %s = %s"%(i,idd,s,con))
