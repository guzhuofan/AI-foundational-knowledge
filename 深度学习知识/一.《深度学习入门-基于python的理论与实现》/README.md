---
date: 2026-05-08
archived: false
---

# 深度学习入门



## 文件结构

|文件夹名   |说明                         |
|:--        |:--                          |
|ch01       |第1章使用的源代码            |
|ch02       |第2章使用的源代码            |
|...        |...                          |
|ch08       |第8章使用的源代码            |
|common     |共同使用的源代码             |
|dataset    |数据集用的源代码             |


源代码的解释请参考本书。

## 必要条件
执行源代码需要按照以下软件。

* Python 3.x
* NumPy
* Matplotlib

※Python的版本为Python 3。

## 执行方法

前进到各章节的文件夹，执行Python命令。

```
$ cd ch01
$ python man.py

$ cd ../ch05
$ python train_nueralnet.py
```

## 对应的虚拟环境名称
`DLpy`

## 如何进入虚拟环境

在终端或命令提示符中运行以下命令：

```bash
conda activate DLpy
```

## 已安装的主要库

该环境已安装最新稳定版的 Python 以及兼容的 Numpy 和 Matplotlib 库：

- **Python**: 3.14.0
- **Numpy**: 2.3.5
- **Matplotlib**: 3.10.6

## 其他依赖

除了上述核心库外，环境还包含运行所需的依赖项（如 mkl, pillow, pyqt 等）。

## 退出环境

如果需要退出当前虚拟环境，请运行：

```bash
conda deactivate
```


