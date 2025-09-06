"""
Implementation of K-nearest neighbours
"""

import pandas as pd
import matplotlib.pyplot as plt

# import knn classifier from sklearn
from sklearn.neighbors import KNeighborsClassifier

#   reading data from csv file
df = pd.read_csv('Datasets/log_reg.csv')

#   plot the data on graph
#plt.scatter(df.age, df.gaming)
#plt.show()

knn = KNeighborsClassifier()

# fit clasified on data. first parameter is given data and second parameter is label or expected value
knn.fit(df[['age']], df.gaming)

# once model is trained. predict label. 
print(knn.predict([[25]]))