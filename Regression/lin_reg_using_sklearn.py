import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import stats
import pickle

df = pd.read_csv('Regression/datasets/lin_reg.csv')

lin_reg = linear_model.LinearRegression()

lin_reg.fit(df[['area']], df.price)

# predict price of a specific area
print(f"Automatically Calculated: {lin_reg.predict([[3000]])}")

# since linear regression uses y= mx + c . so lets manually perform these and check if calculations are correct.

# get value of m
m = lin_reg.coef_

# get value of c
c = lin_reg.intercept_

x = 3000

y = m * x + c

print(f"Manually Calculated: {y}")

plt.scatter(df.area, df.price)
plt.xlabel('Area')
plt.ylabel('Price')
plt.plot(df.area, lin_reg.predict(df[['area']]), color='green')
plt.show()


#   checking if they really are correlated. r value should be closer to 1 or -1
m,c,r,p,err = stats.linregress(df.area, df.price)
print(r)

#   save built model to a binary file
with open('lin_reg_model', 'wb') as f:
    pickle.dump(lin_reg, f)

#   get saved model
with open('lin_reg_model', 'rb') as f:
    mod = pickle.load(f)