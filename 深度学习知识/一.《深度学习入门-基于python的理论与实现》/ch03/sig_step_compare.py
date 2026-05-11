# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

# sigmoid函数
def sigmoid(x):
    # np.exp对每个元素进行指数运算
    # 广播功能对每个元素进行运算
    return 1 / (1 + np.exp(-x))    

# 阶跃函数
def step_function(x):
    # 先将x转换为布尔类型的numpy数组
    # 然后把布尔类型的数组转换为int类型
    # true转换为1, false转换为0
    # 最终效果等同于: 大于0的部分返回1, 小于等于0的部分返回0
    return np.array(x > 0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1)
plt.plot(x, y2, 'k--')
plt.ylim(-0.1, 1.1) #指定图中绘制的y轴的范围
plt.show()
