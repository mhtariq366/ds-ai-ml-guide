from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

digits = load_digits()

print(digits.data[100])

plt.gray()
plt.matshow(digits.images[100])
plt.show()

log_model = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.1)

log_model.fit(X_train, y_train)

print(log_model.score(X_test, y_test))

print(f"Actual Target: {digits.target[100]}")
print(f"Predicted: {log_model.predict([digits.data[100]])}")