"""
Sentiment Analysis on IMDB Dataset, using Multinomial Naive Bayes from sklearn library.
The naive bayes from sklearn is run on imdb dataset. the dataset contains two columns, review, and its label i.e., positive or negative. before running the model, litlle preprocessing is done. 
comments above each line is written to explain the purpose of said lines. 
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#   reading the file
df = pd.read_csv('Datasets/IMDB Dataset.csv')

#   describe sentiment count
print(df.groupby('sentiment').describe())

#   set numerical values 1 and 0 to positive and negative labels respectively
df['label'] = df['sentiment'].apply(lambda x: 1 if 'positive' in x else 0)

#   split dataset into training and testing data. Train is 80% while test is 20%
train_x, test_x, train_y, test_y = train_test_split(df.review, df.label, test_size=0.2)

#   vectorize textual data, to represent them in numbers
vect = CountVectorizer()
train_x_count = vect.fit_transform(train_x)

# print(train_x_count.toarray()[45:50])

#   Call the Multinomial Naive bayes model
nb_model_mn = MultinomialNB()
nb_model_mn.fit(train_x_count, train_y)

test_x_count = vect.transform(test_x)

#   Print the accuracy of model
print(f"Multinomial NB: {nb_model_mn.score(test_x_count, test_y)}")
