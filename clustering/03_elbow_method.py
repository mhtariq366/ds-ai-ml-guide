"""
calculating the appropriate number of clusters to be made in a dataset using elbow method.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Datasets/one_hot.csv')

ss = []

for i in range(1,6):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10, random_state=0)
    km.fit(df[['marks']])
    ss.append(km.inertia_)

plt.plot(range(1,6), ss)
plt.show()


