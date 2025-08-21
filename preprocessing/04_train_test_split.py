"""
Train test split. Using the built in train_test_split library from sklean to split dataset into train and test data.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Datasets/preprocessing01.csv')

X_label = df.iloc[:, :-1].values
y_label = df.iloc[:, -1:].values

X_train, X_test, y_train, y_test = train_test_split(X_label, y_label, test_size=0.2)

print(len(X_train))