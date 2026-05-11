# coding: utf-8
import sys, os
# 获取当前文件的父目录的父目录（即项目根目录）
print(os.path.abspath(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
print(parent_dir)

import numpy as np
from common.functions_note.functions import softmax, cross_entropy_error
from common.gradient_note.gradient import numerical_gradient 

'''
本文件实现了一个简单的神经网络, 并可视化了梯度下降的过程
'''

class simpleNet:
    def __init__(self):
        # 初始化一个网络对象的时候, 随机生成一个2x3的矩阵作为此网络的权重参数
        self.W = np.random.randn(2,3)

    def predict(self, x):
        # x作为输入数据是一个二维向量, 有两个输入, x1和x2
        return np.dot(x, self.W)

    def loss(self, x, t):
        # 计算输入x通过此网络后的输出的z, 是一个3维向量, 每行代表一个样本的输出
        z = self.predict(x)
        # 对z进行softmax归一化处理, 得到输出的概率y, 每行代表一个样本输出的概率
        y = softmax(z)
        # 计算输出y与真实标签t之间的平均交叉熵损失
        loss = cross_entropy_error(y, t)
        return loss

# 我们这里只算一个样本的损失, 先不批量计算
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])
# 实例化一个简单的神经网络对象
net = simpleNet()
# 接下来有点绕:
# f是计算损失函数的函数, 它其实没用到参数w, 而是通过在外部改变net.W的值, 然后调用f
# 用当前的net.W作为网络的参数w, 计算x中每个样本预测的损失并返回平均值
# net, x和t作为函数栈帧G中的变量, 用到了去函数的__global__中查.
f = lambda w: net.loss(x, t)
# 将损失函数f和网络参数net.W作为输入
# 用数值微分计算此时参数net.W应该往哪个方向调整, 才能使损失函数减小
# 方向为之前讨论的, 每个参数计算其偏导
dW = numerical_gradient(f, net.W)
# 这就是我们下一次net.W更新的方向
print(dW)
