import pandas as pd
import numpy as np
from sklearn import linear_model
import math
from scipy import stats

df = pd.read_csv('Datasets/lin_reg_multi_var.csv')
#print(df)

median_bed = math.floor(df['bed'].median())
mean_age = math.floor(df['age'].mean())

df.bed = df.bed.fillna(median_bed)
df.age = df.age.fillna(mean_age)

lin_reg = linear_model.LinearRegression()

lin_reg.fit(df[['area', 'bed', 'age']], df.price)
