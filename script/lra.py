# Linear Regression Algorithm (LRA)

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
#ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# Unit Testing

def create_dataset(amount, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(amount):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

xs, ys = create_dataset(40, 10, 2, correlation='pos')

def slope(xs,ys):
    m = ( (mean(xs) * mean(ys)) - (mean(xs*ys)) ) / ( (mean(xs)*mean(xs)) - (mean(xs*xs)) )
    return m

m = slope(xs,ys)

def y_int(xs,ys):
    b = ( mean(ys) - m*mean(xs) )
    return b

b = y_int(xs,ys)

# creating the squared error and the r^2 value functions

def squared_err(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_err(ys_orig, ys_line)
    squared_err_y = squared_err(ys_orig, y_mean_line)

    return 1 - (squared_error_regr/squared_err_y)


regression_line = [ (m*x) + b for x in xs]

r_squared = coefficient_of_determination(ys, regression_line)
print("coefficient of determination: ", r_squared)

# Testing the Algorithm

predict_x = 7
predict_y = (m*predict_x) + b

plt.scatter(xs,ys)
plt.scatter(predict_x, predict_y, color = 'r')
plt.plot(xs,regression_line)
plt.show()
