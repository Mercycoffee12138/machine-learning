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
# TODO: 算法实现区域（学生填写）
# 目标：实现 BGD 或 SGD 训练线性回归，返回 (w, b, hist)
# 建议：BGD 每轮使用全部样本；SGD 可每次采样一个或小批量样本

def gd_train(X: np.ndarray, y: np.ndarray, lr=0.01, epochs=200, method='SGD', batch_size=32):
    """
    梯度下降训练线性回归模型
    
    Args:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标值，形状为 (n_samples,)
        lr: 学习率
        epochs: 训练轮数
        method: 'BGD' (批量梯度下降) 或 'SGD' (随机梯度下降)
        batch_size: SGD 的批量大小
    
    Returns:
        w: 权重向量，形状为 (n_features,)
        b: 偏置项
        history_mse_list: 每轮的 MSE 历史记录
    """
    n_samples, n_features = X.shape
    
    # 初始化参数
    w = np.random.normal(0, 0.01, n_features)
    b = 0.0
    
    history_mse_list = []
    
    for epoch in range(epochs):
        if method == 'BGD':
            # 批量梯度下降：使用全部样本
            y_pred = X @ w + b
            error = y_pred - y
            
            # 计算梯度
            dw = (1/n_samples) * X.T @ error
            db = (1/n_samples) * np.sum(error)
            
            # 更新参数
            w -= lr * dw
            b -= lr * db
            
        elif method == 'SGD':
            # 随机梯度下降：使用小批量样本
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                # 获取当前批量的索引
                batch_indices = indices[i:i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # 前向传播
                y_pred_batch = X_batch @ w + b
                error_batch = y_pred_batch - y_batch
                
                # 计算梯度
                dw = (1/len(batch_indices)) * X_batch.T @ error_batch
                db = (1/len(batch_indices)) * np.sum(error_batch)
                
                # 更新参数
                w -= lr * dw
                b -= lr * db
        
        # 计算当前轮次的 MSE（在整个训练集上）
        y_pred_full = X @ w + b
        mse = np.mean((y - y_pred_full) ** 2)
        history_mse_list.append(mse)
        
        # 可选：打印进度
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, MSE: {mse:.6f}')
    
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
    # w, b, hist = gd_train(X_train, y_train, lr=0.05, epochs=300)
    #
    # # 评估
    # y_train_pred = X_train @ w + b
    # y_test_pred = X_test @ w + b
    # train_mse = float(np.mean((y_train - y_train_pred) ** 2))
    # test_mse = float(np.mean((y_test - y_test_pred) ** 2))
    # print(f'Train MSE: {train_mse:.4f}')
    # print(f'Test MSE:  {test_mse:.4f}')
    #
    # # 收敛曲线
    # plt.figure(figsize=(6, 4))
    # plt.plot(range(1, len(hist) + 1), hist, label='Train MSE')
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE')
    # plt.title('GD convergence curve')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(FIG_PATH, dpi=150)
    # plt.close()
    # print('Saved figure:', FIG_PATH)


if __name__ == '__main__':
    main()