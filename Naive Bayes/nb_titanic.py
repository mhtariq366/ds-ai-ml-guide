import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('Naive Bayes/datasets/train.csv')

df.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)

labels = df.Survived
features = df.drop(['Survived'], axis='columns')

dummies = pd.get_dummies(features.Sex)

features = pd.concat([features, dummies], axis='columns')
features.drop(['Sex'], inplace=True, axis='columns')

#   check is theres any column with Nan value.
print(f"Is any Nan available: {features.isna().any()}")

#   fill the Nan values with mean value of the column
mean_age = math.floor(features.Age.mean())
features.Age = features.Age.fillna(mean_age)

print(features.isna().any())

print(f"Features length: {len(features)}, Label length: {len(labels)}")

#   split training and testing data
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2)

print(f"Train Features length: {len(train_x)}, Train Label length: {len(train_y)}")
print(f"Test Features length: {len(test_x)}, Test Label length: {len(test_y)}")

nb_model = GaussianNB()

nb_model.fit(train_x, train_y)

#   if you run the model repeatedly, the accuracy score changes. Thats because the data split changes everytime
print(f"Accuracy: {nb_model.score(test_x, test_y)}")
