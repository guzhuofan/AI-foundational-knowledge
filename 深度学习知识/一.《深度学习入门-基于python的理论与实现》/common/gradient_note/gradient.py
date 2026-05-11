# coding: utf-8
import numpy as np

# 处理x为一维数组的情况
def numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad

# 处理x为多个一维数组的情况
def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_1d(f, x)
        
        return grad

# 处理x为任意维度数组的情况
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    # x是网络的参数, 是一个多维数组
    grad = np.zeros_like(x)
    # 构建x的迭代器, 用it存储
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        # 找到当前索引后, 取出当前索引位置上的参数
        tmp_val = x[idx]
        # 给当前索引位置上的参数加上h, 改变了当前网络的参数
        x[idx] = float(tmp_val) + h
        # 求当前网络的损失函数值
        fxh1 = f(x) 
        # 再给x索引上的参数减去h, 改变了当前网络的参数
        x[idx] = tmp_val - h
        # 求当前网络的损失函数值
        fxh2 = f(x)
        # 计算当前参数的偏导数, 更新在grad相同位置中
        grad[idx] = (fxh1 - fxh2) / (2*h)
        # 为迭代器还原值和移动指针部分添加注释说明
        # 还原当前索引上的值为原始值
        x[idx] = tmp_val
        # 移动迭代器到下一个索引
        it.iternext()
        
    return grad