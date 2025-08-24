"""
Scaling the column values into a specific range.
This way two or more columns will have values in a specific range instead of enormously different values.
for example one column has values in tens while the next column has values in thousands. these can cause issues in analysis, so scaling is better. 
"""
import pandas as pd

# a built in scaler from sklearn is imported
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# read file into df
df = pd.read_csv('Datasets/preprocessing01.csv')

# calculate median for filling in missing values
imputer = SimpleImputer(strategy='median')

# filling missing values
imputer = imputer.fit(df[['age']])
df.age = imputer.transform(df[['age']])

imputer = imputer.fit(df[['salary']])
df.salary = imputer.transform(df[['salary']])

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1:]

# scale values in the entire dataset
scale_X_y = StandardScaler()
X = scale_X_y.fit_transform(X)


y = scale_X_y.fit_transform(y)

print(y)
