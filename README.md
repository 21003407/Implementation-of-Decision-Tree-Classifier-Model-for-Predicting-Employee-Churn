Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Moodle-Code Runner
Algorithm
Import the required libraries.
Upload and read the dataset.
Check for any null values using the isnull() function.
From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
Find the accuracy of the model and predict the required values by importing the required module from sklearn.
Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:  A JOANS JAY AUTHERS 
RegisterNumber: 212221240019

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
Output:
Data Head
head

Information:
info

Null dataset:
null

Value_counys():
left

Data Head:
head2

x.head():
xhead

Accuracy:
ss-7

Data Prediction:
predict

Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
