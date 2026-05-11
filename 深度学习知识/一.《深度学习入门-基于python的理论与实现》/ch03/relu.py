# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

# relu函数作为常用的激活函数也是非线性函数
# 神经网络的激活函数必须使用非线性函数, 因为如果使用线性函数作为激活函数, 
# 那么神经网络的层数就没有任何意义了.
# 首先权重和偏移计算是线性运算, 如果激活函数也是线性运算那么不管有多少层,
# 最终输出的都是初始输入的线性组合, n层干的事一层就可以搞定了, 层数没意义了.
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()
