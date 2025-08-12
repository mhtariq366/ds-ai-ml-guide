"""
Sentiment Analysis on IMDB Dataset, using Multinomial Naive Bayes from sklearn library.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#   reading the file
df = pd.read_csv('ds-ai-ml/IMDB Dataset.csv')

# print(df.groupby('sentiment').describe())

df['label'] = df['sentiment'].apply(lambda x: 1 if 'positive' in x else 0)

# print(df.head())

train_x, test_x, train_y, test_y = train_test_split(df.review, df.label, test_size=0.2)

vect = CountVectorizer()
train_x_count = vect.fit_transform(train_x)

# print(train_x_count.toarray()[45:50])

nb_model_mn = MultinomialNB()
nb_model_mn.fit(train_x_count, train_y)

test_x_count = vect.transform(test_x)

print(f"Multinomial NB: {nb_model_mn.score(test_x_count, test_y)}")
