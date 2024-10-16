# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages
2. Load the dataset and prepare the features to load it into the model
3. Build the model
4. Predict results and evaluate performance metrics

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: PRIYAADARSHINI K
RegisterNumber:  212223240126
*/

import pandas as pd
data=pd.read_csv("Salary (2).csv")
data.head()
data.info()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y_test,y_pred)
print(mse)

r2=r2_score(y_test,y_pred)
print(r2)
dt.predict([[5,6]])
```

## Output:
![Screenshot 2024-10-16 124227](https://github.com/user-attachments/assets/bf2e5f6e-f2b0-4af4-9317-1172fbed80f4)

![Screenshot 2024-10-16 124236](https://github.com/user-attachments/assets/112cad82-92c2-4ec7-8ad8-c3340c67df55)

![Screenshot 2024-10-16 124242](https://github.com/user-attachments/assets/4cc7af17-2aa2-4af4-8749-7b722814f8bb)

![Screenshot 2024-10-16 124257](https://github.com/user-attachments/assets/286a2299-2196-428e-b123-7fb38c62fdf2)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
