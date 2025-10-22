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
# 随机梯度下降（SGD）算法实现
def gd_train(X: np.ndarray, y: np.ndarray, lr=0.01, epochs=200, batch_size=1):
    """
    随机梯度下降训练线性回归模型
    
    参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标值，形状为 (n_samples,)
        lr: 学习率
        epochs: 训练轮数
        batch_size: 批量大小，1表示纯SGD，>1表示小批量SGD
    
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
    
    print(f"开始随机梯度下降训练，样本数：{n_samples}，特征数：{n_features}")
    print(f"学习率：{lr}，训练轮数：{epochs}，批量大小：{batch_size}")
    
    for epoch in range(epochs):
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        
        # 按批量处理数据
        for i in range(0, n_samples, batch_size):
            # 获取当前批量的索引
            batch_indices = indices[i:i+batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # 前向传播：计算预测值
            y_pred_batch = X_batch @ w + b
            
            # 计算梯度（基于当前批量）
            error_batch = y_pred_batch - y_batch
            batch_len = len(batch_indices)
            
            # 对权重w的梯度：∂L/∂w = (2/batch_len) * X_batch^T * error_batch
            # 对偏置b的梯度：∂L/∂b = (2/batch_len) * sum(error_batch)
            dw = (2.0 / batch_len) * X_batch.T @ error_batch
            db = (2.0 / batch_len) * np.sum(error_batch)
            
            # 更新参数（随机梯度下降）
            w = w - lr * dw
            b = b - lr * db
        
        # 计算整个训练集上的MSE用于监控收敛
        y_pred_full = X @ w + b
        mse = np.mean((y - y_pred_full) ** 2)
        history_mse_list.append(mse)
        
        # 打印训练进度
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, MSE: {mse:.6f}")
    
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

    # 如已实现，可取消注释进行训练与作图
    w, b, hist = gd_train(X_train, y_train, lr=0.05, epochs=300)
    #
    # # 评估
    y_train_pred = X_train @ w + b
    y_test_pred = X_test @ w + b
    train_mse = float(np.mean((y_train - y_train_pred) ** 2))
    test_mse = float(np.mean((y_test - y_test_pred) ** 2))
    print(f'Train MSE: {train_mse:.4f}')
    print(f'Test MSE:  {test_mse:.4f}')
    #
    # # 收敛曲线
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(hist) + 1), hist, label='Train MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('GD convergence curve')
    plt.legend()
    plt.tight_layout()
    # plt.savefig(FIG_PATH, dpi=150)
    # plt.close()
    # print('Saved figure:', FIG_PATH)


if __name__ == '__main__':
    main()