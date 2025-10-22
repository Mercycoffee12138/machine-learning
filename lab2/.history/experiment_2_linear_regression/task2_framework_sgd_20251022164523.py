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
# TODO: 算法实现区域(学生填写)
# 目标:实现 BGD 或 SGD 训练线性回归,返回 (w, b, hist)
# 建议:BGD 每轮使用全部样本;SGD 可每次采样一个或小批量样本
def gd_train(X: np.ndarray, y: np.ndarray, lr=0.01, epochs=200):
    """
    使用随机梯度下降(SGD)训练线性回归模型
    
    参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标值 (n_samples,)
        lr: 学习率
        epochs: 训练轮数
    
    返回:
        w: 权重向量 (n_features,)
        b: 偏置项
        history_mse_list: 每个epoch的MSE历史记录
    """
    n_samples, n_features = X.shape
    
    # 初始化参数
    w = np.zeros(n_features)
    b = 0.0
    
    # 记录每个epoch的MSE
    history_mse_list = []
    
    # SGD训练
    for epoch in range(epochs):
        # 随机打乱样本顺序
        indices = np.random.permutation(n_samples)
        
        # 对每个样本进行梯度更新
        for idx in indices:
            xi = X[idx]
            yi = y[idx]
            
            # 预测值
            y_pred = np.dot(xi, w) + b
            
            # 计算误差
            error = y_pred - yi
            
            # 更新梯度(SGD每次只用一个样本)
            w -= lr * error * xi
            b -= lr * error
        
        # 每个epoch结束后,计算整个训练集的MSE
        y_pred_all = X @ w + b
        mse = np.mean((y - y_pred_all) ** 2)
        history_mse_list.append(mse)
    
    return w, b, history_mse_list
# ============================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV, sep=';')
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    X_train, X_test = normalize(X_train, X_test)

    print('[Template] Implement gd_train(X_train, y_train) returning w, b, history.')

    # 训练模型
    print('开始训练模型...')
    w, b, hist = gd_train(X_train, y_train, lr=0.05, epochs=300)
    
    # 评估
    y_train_pred = X_train @ w + b
    y_test_pred = X_test @ w + b
    train_mse = float(np.mean((y_train - y_train_pred) ** 2))
    test_mse = float(np.mean((y_test - y_test_pred) ** 2))
    print(f'\n训练完成!')
    print(f'Train MSE: {train_mse:.4f}')
    print(f'Test MSE:  {test_mse:.4f}')
    
    # 收敛曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(hist) + 1), hist, label='Train MSE', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('SGD Convergence Curve (Task 2)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print(f'\n收敛曲线已保存: {FIG_PATH}')


if __name__ == '__main__':
    main()