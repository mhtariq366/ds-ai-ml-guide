"""
KMeans Clustering

This comes under the unsupervised learning. In this we do not have labeled input data for different classes
instead, we only have data and our goal is to distinctly separate data into different cluster based on 
high similarity among intra cluster elements and low similarity among inter clusters.

For this, we need to supply how many clusters, whats the value of k?
Elbow method is used to determine the best value of k.
Calculate error for each number of k. 

"""
import pandas as pd
import matplotlib.pyplot as plt
#   import KMeans from sklearn
from sklearn.cluster import KMeans

#   reading csv file
df = pd.read_csv('Datasets/kmeans_clustering.csv')

#plt.scatter(df['id'],df['salary'])
#plt.show()

#   create a model from sklearn KMeans and assign 3 clusters
km = KMeans(n_clusters=3)

#   predict the 3 clusters
clt = km.fit_predict(df[['id', 'salary']])

#   append the clusters assigned into data frame
df['cluster'] = clt

df0 = df[df.cluster==0]
df1 = df[df.cluster==1]
df2 = df[df.cluster==2]

#   plot three cluster with different colors each to visualize the three clusters made
plt.scatter(df0.id, df0['salary'], color='green')
plt.scatter(df1.id, df1['salary'], color='blue')
plt.scatter(df2.id, df2['salary'], color='red')

plt.show()
