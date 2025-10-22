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
def gd_train(X: np.ndarray, y: np.ndarray, lr=0.001, epochs=200, batch_size=None):
    """
    使用梯度下降训练线性回归模型
    
    参数:
        X: 训练特征矩阵 (n_samples, n_features)
        y: 训练标签 (n_samples,)
        lr: 学习率
        epochs: 训练轮数
        batch_size: 批量大小。None表示BGD(使用全部样本),1表示SGD,其他值表示Mini-batch GD
    
    返回:
        w: 权重向量
        b: 偏置项
        history_mse_list: 每个epoch的MSE历史记录
    """
    n_samples, n_features = X.shape
    
    # 初始化参数
    w = np.zeros(n_features)
    b = 0.0
    
    history_mse_list = []
    
    # 如果batch_size为None,使用BGD(全部样本)
    if batch_size is None:
        batch_size = n_samples
    
    for epoch in range(epochs):
        # 打乱数据索引(用于SGD和Mini-batch GD)
        indices = np.random.permutation(n_samples)
        
        # 按批次训练
        for i in range(0, n_samples, batch_size):
            # 获取当前批次的索引
            batch_indices = indices[i:min(i + batch_size, n_samples)]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # 前向传播:计算预测值
            y_pred = X_batch @ w + b
            
            # 计算梯度
            error = y_pred - y_batch
            dw = (2 / len(batch_indices)) * (X_batch.T @ error)
            db = (2 / len(batch_indices)) * np.sum(error)
            
            # 更新参数
            w -= lr * dw
            b -= lr * db
        
        # 计算整个训练集的MSE(用于监控收敛)
        y_train_pred = X @ w + b
        mse = float(np.mean((y - y_train_pred) ** 2))
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

    print('[Training] Starting gradient descent training...')

    # 训练模型 (可以调整batch_size: None=BGD, 1=SGD, 其他值=Mini-batch GD)
    w, b, hist = gd_train(X_train, y_train, lr=0.05, epochs=300, batch_size=None)

    # 评估
    y_train_pred = X_train @ w + b
    y_test_pred = X_test @ w + b
    train_mse = float(np.mean((y_train - y_train_pred) ** 2))
    test_mse = float(np.mean((y_test - y_test_pred) ** 2))
    print(f'Train MSE: {train_mse:.4f}')
    print(f'Test MSE:  {test_mse:.4f}')

    # 收敛曲线
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(hist) + 1), hist, label='Train MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('GD convergence curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print('Saved figure:', FIG_PATH)


if __name__ == '__main__':
    main()