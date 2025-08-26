import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Datasets/one_hot.csv')

km = KMeans(n_clusters=2)

#plt.scatter(df.subject, df.marks)
#plt.show()

clt = km.fit_predict(df[['subject', 'marks']])

df['clusters'] = clt

df0 = df[df['clusters']==0]
df1 = df[df['clusters']==1]

plt.scatter(df0.subject, df0.marks, color='green')
plt.scatter(df1.subject, df1.marks, color='red')

plt.show()