# Ex.No: 13 Learning – Use Supervised Learning  
### DATE:10-05-2024                                                                           
### REGISTER NUMBER :212221220046 
### AIM: 
To write a program to train the classifier for Diabetes Prediction..
###  Algorithm:
1.Load data, create user-item matrix, calculate user similarity using cosine similarity.
2.Predict ratings for target user based on similar user ratings.
3.Filter out rated books, select top N recommendations with highest predicted ratings.
### Program:
```
import packages
import numpy as np
import pandas as pd

from google.colab import drive
drive.mount('/content/gdrive')

pip install gradio

pip install typing-extensions --upgrade

import gradio as gr

cd /content/gdrive/MyDrive/demo/gradio_project-main

#get the data
data = pd.read_csv('diabetes.csv')
data.head()

print(data.columns)

x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
print(x[:5])

from multi_imbalance.utils.plot import plot_cardinality_and_2d_data
plot_cardinality_and_2d_data(x, y, 'PIMA Diabetes Prediction Data set')

#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y)

#scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

#instatiate model
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy on training set:", model.score(x_train, y_train))
print("Model Accuracy on Test Set:", model.score(x_test, y_test))

print(data.columns)

#create a function for gradio
def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    if(prediction==0):
      return "NO"
    else:
      return "YES"

#create a function for gradio
def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    return prediction

outputs = gr.outputs.Textbox()
app = gr.Interface(fn=diabetes, inputs=['number','number','number','number','number','number','number','number'], outputs=outputs,description="Detection of Diabeties")
app.launch(share=True)
```
### Output:
![image](https://github.com/Roselineb/AI_Lab_2023-24/assets/128909895/34ce5e21-7418-4f61-bcd7-87e8861661fc)
![image](https://github.com/Roselineb/AI_Lab_2023-24/assets/128909895/06191e87-e45c-4c96-8444-bb2541a7c13e)
![image](https://github.com/Roselineb/AI_Lab_2023-24/assets/128909895/884a8414-1fbd-4377-a8fd-dcfe9d9af23d)
### Result:
Thus the system was trained successfully and the prediction was carried out.


