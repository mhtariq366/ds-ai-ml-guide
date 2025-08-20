"""
Linear regression is used for continuous predicted values, while for categorical predicted values,
we use logistic regression.
It can be used for classification problems. Be it binary or multiclass classification.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Datasets/log_reg.csv")

print(df.info())

#plt.scatter(df.age, df.gaming)
#plt.show()

#   by looking at the scatter plot from above lines, we can see that linear reg wont work on this dataset.
#   in logistic regression, we are using sigmoid function: 1/( 1 + e^(-z) )

X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.gaming, test_size=0.2)

log_model = LogisticRegression()

log_model.fit(X_train, y_train)

print(X_test)
print(log_model.predict(X_test))

print(log_model.score(X_test, y_test))