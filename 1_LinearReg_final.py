# -*- coding: utf-8 -*-
"""
@author: Vishnuvardhan Janapati
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model #cross_validation,
 

path='C:/Users/Jeyarani/Desktop/RiskCoder/May2019/Projects/Linear Regression/'
df=pd.read_csv(path + 'ex1data1.txt',header=None)
df.columns=['Population','Profit']

# Inputs (Population and profit in restaurent business)
X=np.array(df['Population'])
y=np.array(df['Profit'])

Lreg=linear_model.LassoCV(eps=0.04,max_iter=1500,tol=1e-4)

Lreg.fit(np.reshape(X,(-1,1)),y)

print('------ Linear Regression------------')
print('Accuracy of Linear Regression Model is ',round(Lreg.score(np.reshape(X,(-1,1)),y)*100,2))

# predicting expected profits if the population is 35000 and 70,000
Predict1=Lreg.predict(np.reshape(3.5,(-1,1)))
Predict2=Lreg.predict(np.reshape(7,(-1,1)))

print('For a population of 35,000 we pridict a profit of ',Predict1*10000)
print('For a population of 70,000 we pridict a profit of ',Predict2*10000)

fig = plt.figure(figsize=(18,9))
a = .2
plt.scatter(X,y,alpha=a)
plt.grid(b=True, which='major', axis='x')
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.title("Populations versus Profit")
plt.grid()