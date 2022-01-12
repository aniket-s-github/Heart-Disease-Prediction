# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 20:36:09 2022

@author: Aniket Dilip Shinde
"""
import numpy as np
import pandas as pd

from logistic_logic import LogisticReg
from logistic_logic import accuracy

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

df=pd.read_csv("heart_disease.csv.txt")
print(df.head())
print(df.info())
# mask1=df['Ca'].isnull()
# mask2=df['Thal'].isnull()
# print(df[mask1|mask2])

df.dropna(inplace=True)
print(df.info())

# create hd array from df values to pre-process data using LabelEncoder
hd=df.iloc[:,:].values
# print(hd)
print(hd.shape)


# convert non-numerical column values to numerical using LabelEncoder
from sklearn.preprocessing import LabelEncoder
lblenco = LabelEncoder()
hd[:,2]=lblenco.fit_transform(hd[:,2])
hd[:,12]=lblenco.fit_transform(hd[:,12])
hd[:,13]=lblenco.fit_transform(hd[:,13])

col_val=['Age','Sex','ChestPain','RestBP','Chol','Fbs','RestECG','MaxHR','ExAng','Oldpeak','Slope','Ca','Thal','AHD']
hd=pd.DataFrame(data=hd, columns=col_val)
# hd['Ca'].fillna(0.0, inplace=True)
print(hd.info())

# filtering hd DataFrame
index=0
while index<14:
    hd.iloc[:,index]=pd.to_numeric(hd.iloc[:,index])
    index+=1
print(hd.info())

# checking correlation between features of hd DataFrame
# print(df[['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Thal']].corr())
print(hd[['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Thal', 'AHD']].corr())

# seperate dataset into features and targets
x=hd.iloc[:,:-1].values
print(x[:6,:])
y=hd.iloc[:,-1].values
print(y[:6])

# applying feature scaling to columns of independent variable (features)
index=0
while index<13:
    max_ele = np.max(x[:,index])
    print(f"max element in col_{index}: ",np.max(x[:,index]))
    x[:,index] = x[:,index]/max_ele
    index+=1
print(x)

f1_test_scores = []
acc_test_scores = []
f1_train_scores = []
acc_train_scores = []
f1_data_scores = []
acc_data_scores = []

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

for train_set, test_set in skf.split(x, y):
    x_train, x_test, y_train, y_test = x[train_set], x[test_set], y[train_set], y[test_set]
    classif = LogisticReg(1, 100)
    classif.fit(x_train, y_train)
    predictions = classif.predict(x_test)
    
    train_predict = classif.predict(x_train)
    data_predict = classif.predict(x)
    
    cm = confusion_matrix(predictions, y_test)
    print(cm)
    
    print('Test F1 score of model: ',f1_score(y_test,predictions))
    print("Test Accuracy of model: ",accuracy(y_test, predictions))
    f1_test_scores.append(f1_score(y_test,predictions))
    acc_test_scores.append(accuracy(y_test, predictions))
    
    print('\nTrain F1 score of model: ',f1_score(y_train,train_predict))
    print("Train Accuracy of model: ",accuracy(y_train, train_predict))
    f1_train_scores.append(f1_score(y_train,train_predict))
    acc_train_scores.append(accuracy(y_train, train_predict))
    
    print('\nData F1 score of model: ',f1_score(y,data_predict))
    print("Data Accuracy of model: ",accuracy(y, data_predict),"\n__________________________________________________\n")
    f1_data_scores.append(f1_score(y,data_predict))
    acc_data_scores.append(accuracy(y, data_predict))
    
print("Test f1 score array: ",np.array(f1_test_scores))
print("Test acc score array: ",np.array(acc_test_scores),"\n")
print("Train f1 score array: ",np.array(f1_train_scores))
print("Train acc score array: ",np.array(acc_train_scores),"\n")
print("Data f1 score array: ",np.array(f1_data_scores))
print("Data acc score array: ",np.array(acc_data_scores))

print("Mean test f1 score: ",np.array(f1_test_scores).mean())
print("Mean test acc score: ",np.array(acc_test_scores).mean(),"\n")
print("Mean train f1 score: ",np.array(f1_train_scores).mean())
print("Mean train acc score: ",np.array(acc_train_scores).mean(),"\n")
print("Mean data f1 score: ",np.array(f1_data_scores).mean())
print("Mean data acc score: ",np.array(acc_data_scores).mean())