# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

'''
本文件实现了任意二元函数function_2的数值微分
'''

# 数值微分求一维变量的梯度
def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    # x是一个一维数组，每个元素都是一个变量
    grad = np.zeros_like(x)
    # 对每个变量进行求导
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) 
        # f(x-h)的计算
        x[idx] = tmp_val - h 
        fxh2 = f(x) 
        grad[idx] = (fxh1 - fxh2) / (2*h)
        # 还原值
        x[idx] = tmp_val
    return grad

# 批量计算多个点的梯度
def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        # X此时是一个二维数组, 每个一维数组代表一个点
        grad = np.zeros_like(X)
        # 对每个点进行求导
        for idx, x in enumerate(X):
            # grad[idx]=... 是将右侧计算出的值实实在在地复制到了grad预分配好的内存块中
            # 而不是像list一样赋值上一个引用
            grad[idx] = _numerical_gradient_no_batch(f, x)
        return grad

# x输入时可能是行向量, 也可能是列向量
def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)
     
if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    
    X = X.flatten()
    Y = Y.flatten()
    
    grad = numerical_gradient(function_2, np.array([X, Y]) )
    
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()