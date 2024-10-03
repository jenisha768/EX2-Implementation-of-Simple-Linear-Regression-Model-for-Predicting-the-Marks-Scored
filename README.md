# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```

Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: JENISHA TEENA ROSE F
RegisterNumber:2305001010

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1.csv')
df.head()
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df[['X']],df['Y'],test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x,lr.predict(x),color='red')
m=lr.coef_
m
b=lr.intercept_
b
pred=lr.predict(x_test)
pred
x_test
y_test
from sklearn.metrics import mean_squared_log_error
mse=mean_squared_log_error(y_test,pred)
mse
```

## Output:
![Screenshot (59)](https://github.com/user-attachments/assets/364b1000-abaa-4c8d-b4a7-01c4312d8397)
![Screenshot (60)](https://github.com/user-attachments/assets/75e31d6c-8ff1-47e9-a1ec-c4417f6e1652)
![Screenshot (61)](https://github.com/user-attachments/assets/639cf4df-c1a4-491b-8ca2-3e84d6441a5c)
![Screenshot (62)](https://github.com/user-attachments/assets/f1a62256-d6e9-46ae-a3c8-33f667adf467)
![Screenshot (63)](https://github.com/user-attachments/assets/b78f1f19-68f4-4992-b35f-67f0928d058c)
![Screenshot (64)](https://github.com/user-attachments/assets/e1d54af2-9c0e-452e-a566-49ee5f35dbe2)









## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
