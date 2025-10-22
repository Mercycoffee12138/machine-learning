#!/usr/bin/env python3
"""
模板（不依赖第三方ML库）：任务4 岭回归（解析法）
- 目的：让学生实现岭回归的闭式解 (X^T X + λI)^{-1} X^T y
- 本模板仅保留数据读取、划分与评估骨架；算法实现留白
- 允许使用 numpy / pandas / matplotlib，禁止使用 sklearn
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
CSV = os.path.join(DATA_DIR, 'winequality-white.csv')

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
# 岭回归闭式解实现
def ridge_fit_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    岭回归闭式解
    
    参数:
        X: 扩展特征矩阵，形状为 (n_samples, n_features + 1)，最后一列为常数1（偏置项）
        y: 目标值，形状为 (n_samples,)
        lam: 正则化参数 λ
    
    返回:
        theta: 参数向量，形状为 (n_features + 1,)，包含权重w和偏置b
    """
    n_features = X.shape[1]
    
    # 创建正则化矩阵 λI
    # 注意：通常不对偏置项进行正则化，所以最后一个对角元素设为0
    I = np.eye(n_features)
    I[-1, -1] = 0  # 不对偏置项进行正则化
    
    # 岭回归闭式解：θ = (X^T X + λI)^(-1) X^T y
    XTX = X.T @ X  # X转置乘以X
    XTX_reg = XTX + lam * I  # 加入正则化项
    XTy = X.T @ y  # X转置乘以y
    
    # 计算逆矩阵并求解
    theta = np.linalg.solve(XTX_reg, XTy)
    
    return theta
# ============================


def main():
    df = pd.read_csv(CSV, sep=';')
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    X_train, X_test = normalize(X_train, X_test)

    print('[Template] Implement ridge_fit_closed_form(X_train, y_train, lam).')

    # 如已实现，可取消注释进行评估
    lam = 1.0
    # # 追加常数列用于偏置
    X_train_ext = np.column_stack([X_train, np.ones(X_train.shape[0])])
    X_test_ext = np.column_stack([X_test, np.ones(X_test.shape[0])])
    theta = ridge_fit_closed_form(X_train_ext, y_train, lam)
    w, b = theta[:-1], float(theta[-1])
    train_mse = mse(y_train, X_train @ w + b)
    test_mse = mse(y_test, X_test @ w + b)
    print(f'lambda={lam} -> Train MSE: {train_mse:.4f}')
    print(f'lambda={lam} -> Test MSE:  {test_mse:.4f}')


if __name__ == '__main__':
    main()