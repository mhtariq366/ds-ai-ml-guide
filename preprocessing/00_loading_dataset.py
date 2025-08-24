"""
The first step is to load the dataset. For this we use pandas library
"""

# import pandas library
import pandas as pd

# using the read csv function from pandas library to load dataset into a data frame
df = pd.read_csv('Datasets/preprocessing01.csv')

# just an example to distribute data into dependent and independent variables
X_label = df.iloc[:, :-1].values
y_label = df.iloc[:, -1:].values

print(y_label)