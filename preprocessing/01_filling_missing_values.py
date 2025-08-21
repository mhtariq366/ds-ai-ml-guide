"""
Often datasets are not having every value against every attribute or column.
Often times there missing values. There are different ways of dealing with missing values.
There is not one specific solution, not a one shoes fits all.
every dataset, every case study has its own preference of solution type.
"""

import pandas as pd

df = pd.read_csv('Datasets/preprocessing01.csv')

#   check if there are any missing values.
print(df.isna().any())


#   run to see missing values, there are shown as 'NaN'
print(df)

"""
Solution 1: Delete/Exclude rows with missing values from analysis
Not an ideal solution. Many important features can be lost.
"""
deleted_df = df.dropna()
print(deleted_df)


"""
Solution 2: Fill the missing values with either mean, mode or median of respective column.

"""
#   using mean value for age
mean_df = df.age.fillna(df.age.mean())
print(mean_df)

#   using median for salary
median_df = df.salary.fillna(df.salary.median())
print(median_df)

