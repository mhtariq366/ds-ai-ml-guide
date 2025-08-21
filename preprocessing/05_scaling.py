"""
Scaling the column values into a specific range.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv('Datasets/preprocessing01.csv')

imputer = SimpleImputer(strategy='median')

imputer = imputer.fit(df[['age']])
df.age = imputer.transform(df[['age']])

imputer = imputer.fit(df[['salary']])
df.salary = imputer.transform(df[['salary']])

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1:]

scale_X_y = StandardScaler()
X = scale_X_y.fit_transform(X)


y = scale_X_y.fit_transform(y)

print(y)
