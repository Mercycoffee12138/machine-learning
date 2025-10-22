import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 数据
k_list = [1, 3, 5, 7, 11, 13, 17]
manual_acc = [91.15, 90.90, 91.15, 91.40, 90.90, 90.21, 89.83]
weka_acc = [91.4626, 90.3327, 90.2699, 90.3327, 90.5838, 89.5794, 89.3911]
kappa = [0.9051, 0.8926, 0.8919, 0.8926, 0.8954, 0.8842, 0.8821]

plt.figure(figsize=(8,5))
plt.plot(k_list, manual_acc, marker='o', label='手动kNN精度')
plt.plot(k_list, weka_acc, marker='s', label='Weka kNN精度')
plt.xlabel('k值')
plt.ylabel('准确率 (%)')
plt.title('手动实现与Weka的kNN性能对比')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(k_list, kappa, marker='^', color='purple', label='混淆熵（Kappa）')
plt.xlabel('k值')
plt.ylabel('Kappa')
plt.title('Weka kNN 混淆熵（Kappa）随k变化')
plt.legend()
plt.grid(True)
plt.show()