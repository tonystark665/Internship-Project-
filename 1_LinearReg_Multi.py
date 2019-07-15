# -*- coding: utf-8 -*-
"""
Problem: Implement a linear Regression with multiple variables to predict the prices of houses.
Training data as 3 columns (Size in square feet, # of bedrooms, the price of the house)
This data set is from the machine learning course by Prof Andrew Ng, Coursera

As the training data is very small (47), the Accuracy of prediction is around 72% (low).
Higher the data size, higher the prediction accuracy. 

@author: Vishnuvardhan Janapati
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation,linear_model
 
# load data
#path=#pwd#'path to the file/'
df=pd.read_csv('ex1data2.txt',header=None)
df.columns=['Size','Bedrooms','Price'] # rename columns


## Inputs (X) and labels (y) (Population and profit in restaurent business)
y=np.array(df['Price'])
X=np.array(df.drop(['Price'],1))
X=X.astype('float64')
Sscaler=preprocessing.StandardScaler()
Xs=Sscaler.fit_transform(X)

# Robust scaler is very helpful in handling outliers
#Rscaler=preprocessing.RobustScaler()
#Xr=Rscaler.fit_transform(X)

# linear regression model
Lreg=linear_model.LassoCV(eps=0.08,max_iter=400,tol=1e-5)
#
Lreg.fit(Xs,y)
#

#print('------ Multivariate Linear Regression------------')
print('Accuracy of Linear Regression Model is ',round(Lreg.score(Xs,y)*100,2))
#
# predicting price of house with 1650 sq. feet in size with 3 bed rooms
Predict1=Lreg.predict(Sscaler.transform(np.reshape([1650,3],(1,-1))))
print('Predicted price of a house with 1650 sq. feet and 3 bed room is $', round(Predict1[0],2))
