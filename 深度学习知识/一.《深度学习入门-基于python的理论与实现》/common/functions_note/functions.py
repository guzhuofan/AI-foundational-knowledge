# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    
# 实现softmax函数也是第三章的一个重点
def softmax(x):
    # 输入的x是批量图片的ndarray时, 需要进行如下操作
    if x.ndim == 2:
        # 假设x的shape为(a, b)
        # 转置后shape变为(b, a)
        x = x.T
        # axis = 0表示聚合b对应的数组, 得到(a,)的数组
        # 然后用广播机制, (b, a) - (a,) = (b, a)
        # 最终效果就是每张图片的每个元素都减去了该图片输出的最大值
        x = x - np.max(x, axis=0)
        # 对x的每个元素进行指数函数运算, 此时数组shape为(b, a)
        # axis = 0表示聚合b对应的数组, 得到(a,)的数组, 每个元素表示该图片输出的和
        # 然后用广播机制, (b, a) / (a,) = (b, a)
        # 最终效果就是每张图片的每个元素都除以了该图片输出的和
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        # 最后转置回(a, b)的shape, 每行表示一张图片的输出
        return y.T 
    # 溢出对策, 实际上就是给每个指数减去一个常数
    # 因为分子分母可以提出exp(-常数), 从而消掉, 不会改变最终结果所以此方法是安全的
    x = x - np.max(x) 
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 这是第四章的重点, 用于计算多个样本的交叉熵误差均值
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 将t中每张图片的one-hot-vector转换成这张图片的标签
    # 最终返回的是每张图片对应的标签的一维数组
    if t.size == y.size:
        t = t.argmax(axis=1)
    # 找出图片个数(batch_size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
