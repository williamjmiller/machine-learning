# Linear Regression Algorithm (LRA)

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

def slope(xs,ys):
    m = ( (mean(xs) * mean(ys)) - (mean(xs*ys)) ) / ( (mean(xs)*mean(xs)) - (mean(xs*xs)) )
    return m

m = slope(xs,ys)

def y_int(xs,ys):
    b = ( mean(ys) - m*mean(xs) )
    return b

b = y_int(xs,ys)

regression_line = [ (m*x) + b for x in xs]

predict_x = 7
predict_y = (m*predict_x) + b

plt.scatter(xs,ys)
plt.scatter(predict_x, predict_y, color = 'r')
plt.plot(xs,regression_line)
plt.show()
