#!/usr/bin/env python3
"""
模板（不依赖第三方ML库）：任务2 梯度下降（BGD/SGD）
- 目的：给学生手写实现线性回归的梯度下降训练（任选 BGD 或 SGD）
- 本模板仅保留数据读取、划分与可视化骨架；算法实现留白
- 允许使用 numpy / pandas / matplotlib，禁止使用 sklearn
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
CSV = os.path.join(DATA_DIR, 'winequality-white.csv')
FIG_PATH = os.path.join(OUT_DIR, 'task2_mse_curve_template.png')

np.random.seed(42)


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio=0.2):
    n = X.shape[0]
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    return X[idx[test_size:]], y[idx[test_size:]], X[idx[:test_size]], y[idx[:test_size]]


def normalize(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std

# ============================
# 批量梯度下降（BGD）算法实现
def gd_train(X: np.ndarray, y: np.ndarray, lr=0.01, epochs=200):
    """
    批量梯度下降训练线性回归模型
    
    参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标值，形状为 (n_samples,)
        lr: 学习率
        epochs: 训练轮数
    
    返回:
        w: 权重向量，形状为 (n_features,)
        b: 偏置项
        history_mse_list: 每轮训练后的MSE历史记录
    """
    n_samples, n_features = X.shape
    
    # 初始化参数
    w = np.random.normal(0, 0.01, n_features)  # 权重初始化为小的随机值
    b = 0.0  # 偏置初始化为0
    
    history_mse_list = []
    
    print(f"开始批量梯度下降训练，样本数：{n_samples}，特征数：{n_features}")
    print(f"学习率：{lr}，训练轮数：{epochs}")
    
    for epoch in range(epochs):
        # 前向传播：计算所有样本的预测值
        y_pred = X @ w + b
        
        # 计算损失（均方误差）
        mse = np.mean((y - y_pred) ** 2)
        history_mse_list.append(mse)
        
        # 计算梯度（基于全部样本）
        error = y_pred - y
        
        # 对权重w的梯度：∂L/∂w = (2/n) * X^T * error
        # 对偏置b的梯度：∂L/∂b = (2/n) * sum(error)
        dw = (2.0 / n_samples) * X.T @ error
        db = (2.0 / n_samples) * np.sum(error)
        
        # 更新参数（批量梯度下降 - 使用全部样本的梯度）
        w = w - lr * dw
        b = b - lr * db
        
        # 打印训练进度
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, MSE: {mse:.6f}")
    
    return w, b, history_mse_list
# ============================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. 读取数据集（分隔符为分号）
    print("正在读取数据集...")
    df = pd.read_csv(CSV, sep=';')
    print(f"数据集形状: {df.shape}")
    print(f"特征列: {df.columns[:-1].tolist()}")
    print(f"目标列: {df.columns[-1]}")
    
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)

    # 2. 按4:1比例划分训练集和测试集（测试集占20%）
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    print(f"训练集样本数: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")
    
    # 数据标准化
    X_train, X_test = normalize(X_train, X_test)
    print("数据标准化完成")

    print('\n开始使用批量梯度下降训练线性回归模型...')

    # 3. 使用批量梯度下降训练模型
    w, b, hist = gd_train(X_train, y_train, lr=0.05, epochs=300)

    # 4. 评估模型性能
    y_train_pred = X_train @ w + b
    y_test_pred = X_test @ w + b
    train_mse = float(np.mean((y_train - y_train_pred) ** 2))
    test_mse = float(np.mean((y_test - y_test_pred) ** 2))
    
    print(f'\n=== 模型评估结果 ===')
    print(f'训练集 MSE: {train_mse:.4f}')
    print(f'测试集 MSE: {test_mse:.4f}')

    # 5. 可视化：绘制MSE收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(hist) + 1), hist, 'b-', linewidth=2, label='训练集 MSE')
    plt.xlabel('迭代次数 (Epoch)')
    plt.ylabel('均方误差 (MSE)')
    plt.title('批量梯度下降 - MSE 收敛曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'\n收敛曲线已保存至: {FIG_PATH}')


if __name__ == '__main__':
    main()