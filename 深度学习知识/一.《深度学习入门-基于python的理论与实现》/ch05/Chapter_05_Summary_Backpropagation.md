---
date: 2026-05-08
archived: true
---

# 第5章 误差反向传播法 (Backpropagation) - 核心笔记

> **Thinking Flow**: "数值微分太慢了 -> 我们需要一种高效计算梯度的方法 -> 误差反向传播法 -> 将复杂的神经网络拆解为简单的'层' -> 像搭积木一样构建高效的网络。"

## 1. 全局视角与脉络回顾 (Global Context & Retrospective)

在进入本章之前，让我们回顾一下你已经走过的路：
*   **Ch03 (神经网络)**: 学习了神经网络的“前向传播”（Forward Propagation），即输入数据，经过权重和激活函数，得到输出。这解决了感知机无法表示复杂函数的问题。
*   **Ch04 (神经网络的学习)**: 引入了“损失函数”来衡量模型好坏，并使用**数值微分**（Numerical Differentiation）来计算梯度，从而更新权重。
    *   *痛点*: 数值微分虽然实现简单、容易理解，但计算量巨大（需要对每个参数计算两次前向传播, +-h的时候需要计算两次loss函数值），在参数动辄百万的深层网络中完全不可用。

**本章 (Ch05) 的使命**: 引入**误差反向传播法**。这是一种能够**高效、快速**计算梯度的方法。如果说 Ch04 是教你“什么是梯度”，Ch05 就是教你“如何在工程实践中算得快”。这是深度学习框架（如 PyTorch, TensorFlow）的核心引擎。

---

## 2. 本章核心叙事 (Chapter Narrative)

作者通过一个非常巧妙的**“计算图” (Computational Graph)** 视角来引入复杂的数学推导。

### 2.1 从直观到抽象：购买苹果
作者没有上来就丢出复杂的链式法则公式，而是用了 `buy_apple.py` 的例子。
*   **问题**: 苹果价格变动 1 元，最终支付金额变动多少？
*   **直观**: 通过计算图，我们可以看到信号是如何**反向**传递的。
*   **关键洞察**: **局部计算**。无论全局多复杂，每个节点只关注自己的输入和输出。这使得我们可以通过组合简单的节点（加法、乘法）来解决极其复杂的问题。

### 2.2 数学引擎：链式法则 (Chain Rule)
反向传播的数学本质就是链式法则。
*   $\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}$
*   这个公式告诉我们：**上游传来的梯度**乘以**当前节点的局部导数**，就是传递给下游的梯度。

### 2.3 乐高积木：层的实现
这是本章最精彩的部分。作者将神经网络的每一个组件（ReLU, Sigmoid, Affine, Softmax）都封装成了一个个独立的**层 (Layer)**。
*   每个层都有 `forward()` 和 `backward()` 方法。
*   **模块化**: 这种设计使得我们可以像搭积木一样构建网络。`TwoLayerNet` (`two_layer_net.py`) 不再需要手写复杂的梯度公式，只需要把层堆叠起来，然后依次调用 `backward()` 即可。

---

## 3. 核心知识点串联 (Core Concepts Weaver)

### 3.1 加法与乘法节点的反向传播
*   **加法节点**: “不做任何改变，原样将梯度传给下一层”。
    *   代码: `layer_naive.py` 中的 `AddLayer`。
    *   直观: $z = x + y$, $\frac{\partial z}{\partial x} = 1$, 所以 `dx = dout * 1`。
*   **乘法节点**: “翻转输入信号乘以梯度”。
    *   代码: `layer_naive.py` 中的 `MulLayer`。
    *   直观: $z = xy$, $\frac{\partial z}{\partial x} = y$, 所以 `dx = dout * y` (注意是乘以另一个输入 $y$)。

### 3.2 Affine 层的矩阵求导 (难点)
这是最容易出错的地方。Affine 层执行的是 $Y = X \cdot W + B$。
*   **维度陷阱**: 在反向传播时，为了使矩阵维度匹配，我们需要使用**转置**。
    *   $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T$
    *   $\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}$
*   **记忆技巧**: 只要盯着矩阵的形状（Shape），拼凑出能相乘的形状，通常就是对的。

### 3.3 Softmax-with-Loss 层的“漂亮结果”
当你把 Softmax 函数（输出层）和 Cross Entropy Error（损失函数）结合在一起时，反向传播的结果出奇地简单：
*   **公式**: $y - t$
*   **含义**: 模型的预测值 ($y$) 与真实标签 ($t$) 之间的差。
*   **本质**: 这不是巧合，而是设计使然。这种“漂亮的结果”意味着梯度的传递非常直接，误差越大，梯度越大，学习越快。

### 3.4 梯度确认 (Gradient Check)
*   **作用**: 数值微分（Ch04）虽然慢，但实现简单且不易出错。反向传播（Ch05）复杂且容易写错 bug。
*   **策略**: 用数值微分的结果来校验反向传播的结果。如果两者极其接近（如 `1e-7` 级别），说明你的反向传播实现是正确的。
*   代码: 参见 `gradient_check.py`。

---

## 4. 你的知识库延伸 (Knowledge Base Integration)

*   **关联 Ch04**: 你的 `train_neuralnet.py` 现在应该使用 `network.gradient()` (反向传播) 而不是 `network.numerical_gradient()`，这将使训练速度提升成百上千倍。
*   **代码结构进化**:
    *   从 `two_layer_net.py` 可以看到，网络变成了 `OrderedDict` 存储的层。这为后面 Ch06、Ch07 引入更复杂的优化器（Optimizer）和卷积层（CNN）打下了基础。
    *   你现在拥有的 `MulLayer`, `AddLayer`, `Relu`, `Affine` 等类，是构建现代深度学习框架的基石。

---
> **Next Step**: 下一章 (Ch06) 我们将利用这些高效的梯度，去探讨如何“更好地”更新参数（SGD, Momentum, Adam），以及如何处理过拟合（Dropout, Batch Normalization）。
