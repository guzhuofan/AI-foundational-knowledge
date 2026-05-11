# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np

# 从网络下载原始数据, 这部分定义了数据的来源和去向
# 数据集的下载地址(这里用的是亚马逊的镜像源)
url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'  # mirror site
# 映射了4个文件的名字(训练图、训练标签、测试图、测试标签)
# .gz是Gzip压缩文件的后缀名
# Python的gzip模块可以直接打开它, 像读普通文件一样读里面的解压后的数据流
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}
# 本文件所在的目录
dataset_dir = os.path.dirname(os.path.abspath(__file__))
# mnist.pkl这个文件最终保存到的路径
save_file = dataset_dir + "/mnist.pkl"
# 训练集的样本数量
train_num = 60000
# 测试集的样本数量
test_num = 10000
# 图像的形状(1, 28, 28)
img_dim = (1, 28, 28)
# 图像的大小(784)
img_size = 784

# 这部分负责从网络下载原始数据
def _download(file_name):
    # 拼接出文件应该所在位置的完整路径
    file_path = dataset_dir + "/" + file_name
    # 如果文件已经存在, 就直接返回
    if os.path.exists(file_path):
        return
    # 如果文件不存在, 就从网络下载
    print("Downloading " + file_name + " ... ")
    # 从网络`url_base + file_name`下载文件, 并保存到指定路径`file_path`
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
# 调用这个函数结合上个函数下载key_file中所有的文件
def download_mnist():
    for v in key_file.values():
       _download(v)
        
# 这部分负责把下载好的代表标签的二进制文件变成Python能认识的NumPy数组
def _load_label(file_name):
    # 拼接出文件应该所在位置的完整路径
    file_path = dataset_dir + "/" + file_name
    # 从文件中读取标签数据, 并转换为NumPy数组
    print("Converting " + file_name + " to NumPy Array ...")
    # gzip.open(file_path, 'rb')返回的f是一个GzipFile对象(类似文件句柄)
    # 它此时还没完全解压所有数据, 只是准备好了文件的遥控器.
    # 现在可以通过f操作这个压缩文件.
    with gzip.open(file_path, 'rb') as f:
            # MNIST 官方说明书里说明了真正的标签数据是从第 8 个字节开始的
            # 所以这里从第 8 个字节开始读取数据
            # np.frombuffer把这段字节流, 每8个bit切一块, 每块解释成一个无符号整数(0-255)
            # 此时的labels就是一个一维数组
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

# 这部分负责把下载好的代表图像的二进制文件变成Python能认识的NumPy数组
def _load_img(file_name):
    # 拼接出文件应该所在位置的完整路径
    file_path = dataset_dir + "/" + file_name
    # 从文件中读取图像数据, 并转换为NumPy数组
    print("Converting " + file_name + " to NumPy Array ...")    
    # gzip.open(file_path, 'rb')返回的f是一个GzipFile对象(类似文件句柄)
    # 它此时还没完全解压所有数据, 只是准备好了文件的遥控器.
    # 现在可以通过f操作这个压缩文件.
    with gzip.open(file_path, 'rb') as f:
            # 当你调用f.read()时, 他开始读取磁盘上的压缩数据
            # 在内存中进行解压, 返回一个巨大的bytes对象(字节串)
            # 图像数据是从第16个字节开始的所以偏移量选择16, 从第16个字节开始读取数据
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            # np.frombuffer把这段字节流, 每8个bit切一块, 每块解释成一个无符号整数(0-255)
            # 此时的data就是一个一维数组
    # frombuffer默认只会生成一维数组, 所以必须要有下一步的reshape
    # -1的含义是自动推导, 这里已知每个图片784个像素, 所以reshape为(-1, 784)
    # 即自动推导为60000(训练集样本数)或10000(测试集样本数)个样本
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
# 所以综上我们可以看出我们下载的四个压缩文件中, 本质上就是一长串连在一起的整数(字节)
# 全靠我们(程序员)根据"说明书"(前16字节是头, 每784字节是一张图), 拿着剪刀(reshape)把它剪开,拼好

# 创建字典存储4个处理好的NumPy数组
def _convert_numpy():
    # 先创建一个空字典, 用来存储4个NumPy数组
    dataset = {}
    # 调用_load_img加载训练图像的ndarray, 并存储到dataset['train_img']
    dataset['train_img'] =  _load_img(key_file['train_img'])
    # 调用_load_label加载训练标签的ndarray, 并存储到dataset['train_label']
    dataset['train_label'] = _load_label(key_file['train_label'])    
    # 调用_load_img加载测试图像的ndarray, 并存储到dataset['test_img']
    dataset['test_img'] = _load_img(key_file['test_img'])
    # 调用_load_label加载测试标签的ndarray, 并存储到dataset['test_label']
    dataset['test_label'] = _load_label(key_file['test_label'])
    return dataset

# 这部分负责把上面的NumPy数组打包成一个pickle文件
def init_mnist():
    # 下载4个压缩包
    download_mnist()
    # 调用_convert_numpy处理下载好的文件, 并返回一个字典
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    # 以二进制写入模式('wb')打开或创建save_file文件
    # 以f作为文件即mnist.pkl的句柄
    # 并将dataset字典序列化(打包)到mnist.pkl文件中
    # -1表示使用最高版本的Pickle协议(效率最高)
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

# 这部分负责把标签从整数转换为one-hot编码
def _change_one_hot_label(X):
    # 先创建一个全0的数组, 形状为(样本数, 10)
    # 这里10是因为MNIST数据集有10个类别(0-9)
    T = np.zeros((X.size, 10))
    # 遍历每个样本
    # idx是当前样本的索引(行号), row是当前样本的one-hot编码数组
    for idx, row in enumerate(T):
        # 将该样本的标签对应的位置设为1
        # 例如, 如果标签是2, 则row[2] = 1   
        row[X[idx]] = 1
    return T
    
# 这部分负责加载pickle文件, 并返回4个NumPy数组
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指将标签2转换为[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    # 如果pickle文件不存在, 则调用init_mnist初始化数据集
    if not os.path.exists(save_file):
        init_mnist()
    
    # 以二进制读取模式('rb')打开mnist.pkl文件
    # 以f作为文件即mnist.pkl的句柄
    # 并将mnist.pkl文件反序列化(解包)到dataset字典中
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # 如果normalize为True, 则将图像数据转换为float32类型, 并归一化到0.0~1.0
    if normalize:
        for key in ('train_img', 'test_img'):
            # 先将ndarray中的每个元素转化为float32类型
            dataset[key] = dataset[key].astype(np.float32)
            # 然后给每个元素除以255.0, 实现归一化到0.0~1.0
            # 用到了广播机制
            dataset[key] /= 255.0
    
    # 如果one_hot_label为True, 则将标签转换为one-hot编码
    if one_hot_label:
        # 输入的是标签的一维数组, 输出的是one-hot编码的二维数组
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    # 如果flatten为False, 则维持为最初的二维数组(每个样本是784维向量)   
    if not flatten:
         for key in ('train_img', 'test_img'):
            # 本来是n*784的二维数组, 现在变成n*1*28*28的四维数组
            # 这里1是通道数, 28*28是图片的高宽
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


if __name__ == '__main__':
    init_mnist()
