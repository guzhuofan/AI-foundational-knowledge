import numpy as np

print("=== 1. 最基础用法：单纯遍历值 (就像 for x in list) ===")
a = np.arange(6).reshape(2, 3)
print(f"数组 a:\n{a}")
print("遍历结果:", end=" ")
for x in np.nditer(a):
    print(x, end=" ")
print("\n")


print("=== 2. 进阶用法：一边遍历一边修改 (readwrite) ===")
# 默认情况下 nditer 是只读的，要修改必须加 op_flags
with np.nditer(a, op_flags=['readwrite']) as it:
    for x in it:
        x[...] = x * 2  # 注意：必须用 x[...] 来修改内容
print(f"翻倍后的 a:\n{a}")
print("\n")


print("=== 3. 你的用法：需要知道坐标 (flags=['multi_index']) ===")
# 当我们需要知道“我现在在哪个位置”时，必须开启这个 flag
b = np.array([[10, 20], [30, 40]])
it = np.nditer(b, flags=['multi_index'])
while not it.finished:
    idx = it.multi_index
    val = b[idx]
    print(f"坐标: {idx} -> 值: {val}")
    it.iternext()
print("\n")


print("=== 4. 高级用法：同时遍历两个数组 (自动广播) ===")
# 这是 nditer 极其强大的地方，它可以自动处理形状广播
# 比如：让一个 (3, 1) 的数组 和 (1, 3) 的数组一起遍历
x = np.arange(3).reshape(3, 1)  # 0, 1, 2 (竖着)
y = np.arange(3).reshape(1, 3)  # 0, 1, 2 (横着)
print(f"数组 x (3,1):\n{x}")
print(f"数组 y (1,3):\n{y}")

print("开始同步遍历 (自动广播成 3x3):")
for a_val, b_val in np.nditer([x, y]):
    print(f"{a_val} + {b_val} = {a_val + b_val}")
