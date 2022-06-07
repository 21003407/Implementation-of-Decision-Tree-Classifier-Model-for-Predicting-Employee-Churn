# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload the csv file and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree inport DecisionTreeRegressor.
5. Import metrics and calculate the Mean squared error.
6. Apply metrics to the dataset, and predict the output.


## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: P.Lakshmi priya
RegisterNumber:  212221230053
```
```
import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position","Level"]]
y = data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2 = metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
### Data Head:
![head](https://user-images.githubusercontent.com/93427923/169694235-41a469cc-ff3e-4c56-b36c-029319ef1f94.png)

### Data Info:
![info](https://user-images.githubusercontent.com/93427923/169694238-85077655-4a64-4334-b451-997c7ea1937d.png)

### Data Head after applying LabelEncoder():
![head2](https://user-images.githubusercontent.com/93427923/169694242-dd7cae7b-50db-4864-96aa-ca8eb07514e3.png)

### MSE:
![mse](https://user-images.githubusercontent.com/93427923/169694248-eefed989-8fc7-4e80-b3af-992667d1936a.png)

### r2:
![r2](https://user-images.githubusercontent.com/93427923/169694252-b17fc5dd-22fd-46e0-b8de-991fd12528ed.png)

### Data Prediction:
![predict](https://user-images.githubusercontent.com/93427923/169694255-16669af0-0ed0-416e-b387-d63f2f3e9dc3.png)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
