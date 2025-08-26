import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('Datasets/log_reg.csv')

#plt.scatter(df.age, df.gaming)
#plt.show()

knn = KNeighborsClassifier()

knn.fit(df[['age']], df.gaming)

print(knn.predict([[25]]))