import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_logic import LogisticReg
from logistic_logic import accuracy

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

def data_analyse_process():
    df=pd.read_csv("heart_disease.csv.txt")
    print(df.head())
    print(df.info())
    # mask1=df['Ca'].isnull()
    # mask2=df['Thal'].isnull()
    # print(df[mask1|mask2])
    
    df.dropna(inplace=True)
    print(df.info(),"\n")
    
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
    print(hd.info(),"\n")
    
    # checking correlation between features of hd DataFrame
    # print(df[['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Thal']].corr())
    print(hd[['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Thal', 'AHD']].corr())
    
    # seperate dataset into features and targets
    global x,y
    x=hd.iloc[:,:-1].values
    print(x[:6,:])
    y=hd.iloc[:,-1].values
    print(y[:6],"\n")
    
    # applying feature scaling to columns of independent variable (features)
    index=0
    while index<13:
        max_ele = np.max(x[:,index])
        print(f"max element in col_{index}: ",np.max(x[:,index]))
        x[:,index] = x[:,index]/max_ele
        index+=1
    print("\n",x,"\n")

data_analyse_process()

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, stratify=y, random_state=123)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# count 1 & 0 in whole dataset
count_y_1=0
count_y_0=0
for i in y:
    if i==0:
        count_y_0+=1
    else:
        count_y_1+=1

# count 1 & 0 in training dataset
count_train_1=0
count_train_0=0
for i in y_train:
    if i==0:
        count_train_0+=1
    else:
        count_train_1+=1
bar_x1=[1,2]
bar_y1=[count_train_0,count_train_1]

# count 1 & 0 in test dataset
count_test_1=0
count_test_0=0
for i in y_test:
    if i==0:
        count_test_0+=1
    else:
        count_test_1+=1
bar_x2=[1,2]
bar_y2=[count_test_0,count_test_1]

print('Class_0 train points: ',100*count_train_0/count_y_0,'%')
print('Class_1 train points: ',100*count_train_1/count_y_1,'%')
plt.bar(bar_x1,bar_y1, tick_label=[0,1], width=0.5, color=['green','blue'])
plt.title('y_train classes')
plt.show()

print('\nClass_0 test points: ',100*count_test_0/count_y_0,'%')
print('Class_1 test points: ',100*count_test_1/count_y_1,'%')
plt.bar(bar_x2,bar_y2, tick_label=[0,1], width=0.5, color=['green','blue'])
plt.title('y_test classes')
plt.show()

classif = LogisticReg(1, 100)
classif.fit(x_train, y_train)
predictions = classif.predict(x_test)

train_predict = classif.predict(x_train)
data_predict = classif.predict(x)

print("\n",predictions[0:10])
print(" ",y_test[0:10])

cm = confusion_matrix(predictions, y_test)
print(cm)

print('Test F1 score of model: ',f1_score(y_test,predictions))
print("Test Accuracy of model: ",accuracy(y_test, predictions))

print('\nTrain F1 score of model: ',f1_score(y_train,train_predict))
print("Train Accuracy of model: ",accuracy(y_train, train_predict))

print('\nData F1 score of model: ',f1_score(y,data_predict))
print("Data Accuracy of model: ",accuracy(y, data_predict))