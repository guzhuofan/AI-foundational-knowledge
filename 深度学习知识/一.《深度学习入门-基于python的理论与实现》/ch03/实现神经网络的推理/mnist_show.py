# coding: utf-8
import sys, os
print(os.path.abspath(__file__))
print(os.path.dirname(os.path.abspath(__file__)))
# d:\桌面\顾卓凡的知识库\AI知识学习\深度学习知识\一.《深度学习入门-基于python的理论与实现》
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 将上面这个路径添加到sys.path中, 以便我们可以导入这个目录下的文件
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    """显示图片"""
    # np.uint8(img):
    # 它会把img这个ndarray里的每个元素都转换为8位无符号整数, 范围是0~255
    # 详细说就是把输入的ndarray中每个元素的值都转为内存中8bit的数字, 溢出的直接截断
    # 假如一个元素是1 0000 0001, 其之后会变为0000 0001
    
    # Image.fromarray(...): 
    # 它的输入是一个Numpy数组, 输出是一个Image对象, 
    # 它扫描数组里的数字, 把每个数字当作一个像素的亮度, 存在一个图像对象里
    # 对这个对象调用show()方法, 就可以在屏幕上显示出这个图片
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 调用 load_mnist 获取数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 取第一张训练图片
img = x_train[0]
label = t_train[0]
print("img长这样:", img) # 784个像素值的一维数组
print("label长这样:", label)  # 5

print(img.shape)  # (784,)
# 把图像的形状变为原来的尺寸, 需要清楚reshape的参数顺序不要让像素点的顺序被改变.
# 内存层面(物理真实): 每个ndarray对象的数据都是一段连续的, 线性的内存块.
# 视图层面(逻辑抽象): ndarray对象的shape(形状)和strides(步幅)属性决定了对象怎么看这段内存.
# 节省内存: 多个不同形状的数组可以共享同一块内存数据, reshape操作只需要重新创建一个ndarray对象, 给其赋上新的shape和strides属性就完成了数组形状重塑.
# Flatten (拉直) :
# 加入原本形状为 [[1, 2], [3, 4]] (2x2)
# 拉直时, Numpy默认按行优先(Row-Major)顺序, 按内存中顺序读完作为一行.
# 结果为 [1, 2, 3, 4]
# Reshape(还原):
# 当你把[1, 2, 3, 4]reshape成(2, 2)时.
# 它还是按内存从左到右的读取顺序填, 先把 1, 2 填入第一行, 再把 3, 4 填入第二行.
# 结果完美还原为[[1, 2], [3, 4]] 
img = img.reshape(28, 28)  
# img现在是28x28的二维数组
print(img.shape)  # (28, 28)
print(img)

# 显示图片
img_show(img)
