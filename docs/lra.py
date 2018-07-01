# Linear Regression Algorithm (LRA)

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([6, 5, 6, 7, 9, 3], dtype=np.float64)

def slope(xs,ys):
    m = ( (mean(xs) * mean(ys)) - (mean(xs*ys)) ) / ( (mean(xs)*mean(xs)) - (mean(xs*xs)) )
    return m

m = slope(xs,ys)

print(m)
