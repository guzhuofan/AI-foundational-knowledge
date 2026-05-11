# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

'''
本文件实现了任意一元函数function_1的数值微分及任意一点切线函数的绘制
'''

# 中心差分实现的一元函数数值微分
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# 被求导的函数
def function_1(x):
    return 0.01*x**2 + 0.1*x 

# 返回切线函数
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
x = np.arange(0.0, 20.0, 0.1)
y1 = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 15)
y2 = tf(x)

plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
