# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient

'''
本文件实现了函数的梯度下降法, 并可视化了梯度下降的过程
'''

# 对f进行梯度下降, 记录x的中途位置, x必须为一维向量
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []
    # 进行step_num次梯度下降
    for _ in range(step_num):
        x_history.append( x.copy() )
        # 计算当前点的梯度
        grad = numerical_gradient(f, x)
        # 更新x, 朝负梯度方向走一丢丢
        x -= lr * grad
    # 最后一步下降后, 再将x添加到历史记录中
    x_history.append( x.copy() )
    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
