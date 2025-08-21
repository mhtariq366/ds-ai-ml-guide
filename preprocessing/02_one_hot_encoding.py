"""
There are different ways of implementing one hot encoding, we ll discuss them one by one.
"""
import pandas as pd
from sklearn import linear_model

df = pd.read_csv('Datasets/one_hot.csv')

dummies = pd.get_dummies(df.name)

final_df = pd.concat([df, dummies], axis='columns')
final_df = final_df.drop(['name', 'bob'], axis='columns')

print(final_df)

lin_reg = linear_model.LinearRegression()

x = final_df.drop('marks', axis='columns')
y = final_df['marks']

lin_reg.fit(x,y)

print(lin_reg.predict([[1,1,0]]))

print(lin_reg.score(x,y))
