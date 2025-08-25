"""
We have seen linear regression with using only one variable. What if the prediction is based on multiple variables?
Lets says the price of an apartment depends on the area of the apartment, total no. of beds and the age of the building.
These three variables determine the total predicted price of the apartment.
Multi variable linear regression like this example is explored below.
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
import math
from scipy import stats

df = pd.read_csv('Datasets/lin_reg_multi_var.csv')
#print(df)

#   calculate median beds, and mean age of building to fill in the missing values
median_bed = math.floor(df['bed'].median())
mean_age = math.floor(df['age'].mean())

#   filling missing values
df.bed = df.bed.fillna(median_bed)
df.age = df.age.fillna(mean_age)

#   get linear model
lin_reg = linear_model.LinearRegression()

#   fit linear model with three variable area, bed and age
lin_reg.fit(df[['area', 'bed', 'age']], df.price)

