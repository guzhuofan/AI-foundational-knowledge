# type(对象)就相当于查看实例对象的class属性
a = 1
print(a.__class__)

import numpy as np
a = np.array([[1,2], [3,4]])
# 测试广播机制
# 好理解的情况
b = np.array([1,2])
print(a+b)
# 不确定可不可以广播的情况
c = np.array([[1],[2]])
print(a+c)
# 竟然也可以竖着广播!?

# np模块提供了很多工厂方法来生产ndarray数组
# 例如，arange方法也可以生成一个数组，它的元素是一个范围
d = np.arange(4, 10)
print(d)
print(type(d))

print(d>5)
# ndarray也可以作为一个索引数组
e = d[d>5]
print(e)
print(e.__class__)