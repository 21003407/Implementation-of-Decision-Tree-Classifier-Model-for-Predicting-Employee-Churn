# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.    
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: HARSHAVARDHINI M
RegisterNumber: 212221240015 

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evalution","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Data Head
![head](https://user-images.githubusercontent.com/93427208/169464817-681d5776-e0a4-415a-bc58-10e5a9f37cc2.png)

### Information:
![info](https://user-images.githubusercontent.com/93427208/169464890-1a6c35c0-c7e5-43c8-9a40-2c6a0ee9a27b.png)

### Null dataset:
![null](https://user-images.githubusercontent.com/93427208/169464953-b46cc08c-2005-4acf-8fe7-08cae47aa60c.png)

### Value_counys():
![left](https://user-images.githubusercontent.com/93427208/169465015-f4c16c0c-4ca2-4fca-8a7a-f85cb423a999.png)

### Data Head:
![head2](https://user-images.githubusercontent.com/93427208/169465081-dc868d4c-f879-42c1-9be5-8a7c690af454.png)

### x.head():
![xhead](https://user-images.githubusercontent.com/93427208/169465176-9b8d266d-ed47-4a8e-8781-19c04a0360c0.png)

### Accuracy:
![ss-7](https://user-images.githubusercontent.com/93427208/169465389-51117326-7f9b-416e-b48b-a43961d7f513.png)

### Data Prediction:
![predict](https://user-images.githubusercontent.com/93427208/169465493-62ba0d4e-3deb-44ec-a2f1-bd2707cd9b5c.png)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
