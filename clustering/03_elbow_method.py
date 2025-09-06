"""
For clustering, one very important aspect is to determine how many clusters needs to be made.
Some types of data may gather into two clusters, some in three and so on and so forth.
To carefully find out the number of clusters that are appropriate to divide the data properly, a method called elbow method is used.

Below is the implementation of calculating the number of clusters to be made in a dataset using elbow method.
"""ß

import pandas as pd
import matplotlib.pyplot as plt

#   import the kmeans cluster module
from sklearn.cluster import KMeans

#   reading the csv dataset
df = pd.read_csv('Datasets/one_hot.csv')

ss = []

for i in range(1,6):        # six iterations to find the number of clusters. Can be as much iteration as you want
    km = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10, random_state=0)
    km.fit(df[['marks']])
    ss.append(km.inertia_)


#   display the elbow method curve
plt.plot(range(1,6), ss)
plt.show()


