# coding: utf-8
import sys, os
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)  # 为了导入文件而进行的设定, path为D:\桌面\顾卓凡的知识库\AI知识学习\深度学习知识\一.《深度学习入门-基于python的理论与实现》
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 初始化神经网络的权重参数
def init_network():
    with open(path + "/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

# 构建推理函数
# 本章我们的重点在于理解神经网络的结构和定义, 以及神经网络解决实际问题时, 内部是怎么运作的
# 重点在于体会神经网络不同于感知机的激活函数, 和矩阵计算的过程中细节的理解.
# 下一章我们的重点在于构造好的神经网络如何从数据中自动学习权重参数.
# 重点在于理解学习的原理和为什么神经网络学习数据的时候需要使用不同的激活函数.
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))