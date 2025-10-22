#!/usr/bin/env python3
"""
模板（不依赖第三方ML库）：任务3 学习率曲线（基于手写GD）
- 目的：让学生实现线性回归的梯度下降训练函数，并比较不同学习率下的收敛曲线
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
FIG_PATH = os.path.join(OUT_DIR, 'task3_lr_curves_template.png')

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


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

# ============================
# TODO: 算法实现区域（学生填写）
# 目标：实现梯度下降训练线性回归，返回 (w, b, hist)
# 建议：BGD 每轮使用全部样本；可扩展为小批量 SGD
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
    df = pd.read_csv(CSV, sep=';')
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    X_train, X_test = normalize(X_train, X_test)

    print('[Template] Implement gd_train(X_train, y_train) to compare learning rates.')

    # 如已实现，可取消注释进行多学习率对比与作图
    lrs = [0.001, 0.01, 0.05, 0.1]
    curves = {}
    epochs = 300
    for lr in lrs:
        w, b, hist = gd_train(X_train, y_train, lr=lr, epochs=epochs)
        train_m = mse(y_train, X_train @ w + b)
        test_m = mse(y_test, X_test @ w + b)
        curves[lr] = {'hist': hist, 'train_mse': train_m, 'test_mse': test_m}
        print(f'lr={lr:.3f} -> Train MSE={train_m:.4f}, Test MSE={test_m:.4f}')
    #
    plt.figure(figsize=(7, 5))
    for lr, info in curves.items():
        plt.plot(range(1, len(info['hist']) + 1), info['hist'], label=f'lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Train MSE')
    plt.title('MSE convergence under different learning rates')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print('Saved figure:', FIG_PATH)


if __name__ == '__main__':
    main()