"""
Apart from one hot encoding, another way to convert the data into numerical format is label encoder.
we'll discuss it below.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Datasets/preprocessing01.csv')

print(df)

#   as can be seen from output. Column name 'city' needs to be converted to numerical values

label_city = LabelEncoder()
df.city = label_city.fit_transform(df[['city']])

print(df)