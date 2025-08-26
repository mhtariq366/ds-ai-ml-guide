"""
KMeans Clustering

This comes under the unsupervised learning. In this we do not have labeled input data for different classes
instead, we just have data and our goal is to distinctly separate this data into different cluster based on 
high similarity among intra cluster elements and low similarity among inter clusters.

For this, we need to supply how many clusters, whats the value of k?
Elbow method is used to determine the best value of k.
Calculate error for each number of k. And we can see which value of k is appropriate.

When k is choosen. Assign centroids. Distance for each point will be calculated from the centroids.
Simply explained:
1. choose centroids
2. draw straight line between them.
3. draw straight perpendicular line to line in step 2.
4. all points on one side of centroid belongs to that centroid. and other side points belong to other centroids.
5. repeat steps 1-4 until elements are change clusters.
6. Once elements stop changing clusters assigned. stop the algorithm.

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
