# coding: utf-8
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

# 使用绝对路径，确保无论在哪运行脚本都能找到图片
# 获取当前脚本(__file__)所在的目录
print(os.path.dirname(__file__))
# 然后去拼接相对路径
img_path = os.path.join(os.path.dirname(__file__), '../dataset/lena.png')
print(img_path)

# 读入图像
img = imread(img_path) 
plt.imshow(img)

# 显示图像
plt.show()