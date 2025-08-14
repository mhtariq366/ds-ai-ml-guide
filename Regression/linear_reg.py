"""
basic linear regression example
using stats.linregress from scipy.
this is simpler method to implement linear regression.
the functions takes two 1D arrays and returns 5 values.

Prediction on new data can be done manually using the mx+c formula.

for Machine Learning based applications. preferably use the linear regression from linear model in sklearn. 
its done in file named lin_reg.py in the same folder. 
"""
import matplotlib.pyplot as plt
import random
from scipy import stats

# function get values for y
def get_y(x):
    return m * x + c

# two random generated lists values for x and y
x = [random.randint(1,20) for _ in range(10)]
y = [random.randint(1,200) for _ in range(10)]

# get 5 values from stats
m, c, r, p, err = stats.linregress(x, y)

# m is slope
# c is y intercept
# r is the value of relationship. 0 is no relation while 1 and -1 means correlated dataset
# p is value for hypothesis test
# err is standard error for slope

lin_model = list(map(get_y, x))

# plot a graph using matplotlib
plt.scatter(x,y)
plt.plot(x, lin_model)
plt.show()

# print to check values. testing. 
print(f"m: {m}")
print(f"c: {c}")
print(f"r: {r}")
print(f"p: {p}")
print(f"err: {err}")

