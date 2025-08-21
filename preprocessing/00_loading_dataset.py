"""
The first step is to load the dataset. For this we use pandas library
"""

import pandas as pd

df = pd.read_csv('Datasets/preprocessing01.csv')

X_label = df.iloc[:, :-1].values
y_label = df.iloc[:, -1:].values

print(y_label)