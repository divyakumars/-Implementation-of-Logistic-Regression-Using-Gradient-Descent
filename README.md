# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 
STEP 1. Start the program.

STEP 2. Data Preprocessing: Read dataset, drop unnecessary columns, and encode categorical variables.

STEP 3. Initialize Parameters: Initialize theta randomly and extract features (x) and target variable (y).

STEP 4. Define Sigmoid Function: Implement the sigmoid function to transform linear outputs into probabilities.

STEP 5. Define Loss Function and Gradient Descent: Define loss function using sigmoid output and implement gradient descent to minimize loss.

STEP 6. Prediction and Evaluation: Use trained parameters to predict on dataset, calculate accuracy, and optionally predict placement status of new data points.

STEP 7. End the program.


## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: DIVYA K
RegisterNumber:  212222230035
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Placement_Data (1).csv')  
data

data=data.drop('sl_no',axis=1)
data=data.drop('salary',axis=1)

data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data.dtypes

data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
y

theta = np.random.randn(x.shape[1])
Y=y

def sigmoid(z):
  return 1/(1+np.exp(-z))


def loss(theta,X,y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))


def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient = X.T.dot(h-y)/m
    theta-=alpha * gradient
  return theta
theta=gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred 

y_pred = predict(theta,x)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy: ",accuracy)
print(y_pred)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

```

## Output:

![image](https://github.com/user-attachments/assets/06d6c26b-c926-49f3-ac59-1cafb6f03b74)
![image](https://github.com/user-attachments/assets/ee15f87b-45dc-43c8-8ae7-62074d08d9ff)

![image](https://github.com/user-attachments/assets/74143902-1892-487f-b724-22dde21a7c18)
![image](https://github.com/user-attachments/assets/ecbba56d-a0e4-4c38-9702-2f4e9459980f)
![image](https://github.com/user-attachments/assets/549537eb-5f99-4237-a110-36c94decbfb7)

![image](https://github.com/user-attachments/assets/46e73d75-cfec-4a9e-b831-00eaae5fde36)


![image](https://github.com/user-attachments/assets/1e5c231c-9212-484c-a5b7-c3f8169ab0c1)

![image](https://github.com/user-attachments/assets/47b8e46c-ea0d-4653-828b-1183a6bafc8c)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

