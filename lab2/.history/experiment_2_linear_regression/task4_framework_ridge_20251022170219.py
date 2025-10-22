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
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 读取数据集
    print("正在读取红酒质量数据集...")
    df = pd.read_csv(CSV, sep=';')
    print(f"数据集形状: {df.shape}")
    print(f"特征列: {df.columns[:-1].tolist()}")
    print(f"目标列: {df.columns[-1]}")
    
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)

    # 按4:1比例划分训练集和测试集
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    print(f"\n训练集样本数: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")
    print(f"特征数: {X_train.shape[1]}")
    
    # 数据标准化
    X_train, X_test = normalize(X_train, X_test)
    print("数据标准化完成")

    print('\n=== 岭回归正则化参数分析 ===')

    # 测试不同的正则化参数（从很小到很大的范围）
    lambdas = np.logspace(-4, 3, 20)  # 从0.0001到1000，20个点
    lambdas = np.concatenate([[0.0], lambdas])  # 包含λ=0（普通线性回归）
    
    train_mses = []
    test_mses = []
    weight_norms = []  # 权重的L2范数
    weight_maxs = []   # 权重的最大绝对值
    all_weights = []   # 存储所有权重向量用于可视化
    
    print("λ值\t\t训练MSE\t\t测试MSE\t\t权重L2范数\t权重最大值")
    print("-" * 70)
    
    for lam in lambdas:
        # 追加常数列用于偏置项
        X_train_ext = np.column_stack([X_train, np.ones(X_train.shape[0])])
        X_test_ext = np.column_stack([X_test, np.ones(X_test.shape[0])])
        
        # 使用岭回归闭式解训练模型
        theta = ridge_fit_closed_form(X_train_ext, y_train, lam)
        w, b = theta[:-1], float(theta[-1])
        all_weights.append(w)
        
        # 计算MSE
        train_mse = mse(y_train, X_train @ w + b)
        test_mse = mse(y_test, X_test @ w + b)
        
        # 计算权重统计信息
        weight_norm = np.linalg.norm(w)  # L2范数
        weight_max = np.max(np.abs(w))   # 最大绝对值
        
        train_mses.append(train_mse)
        test_mses.append(test_mse)
        weight_norms.append(weight_norm)
        weight_maxs.append(weight_max)
        
        # 打印结果
        print(f"{lam:.4e}\t{train_mse:.4f}\t\t{test_mse:.4f}\t\t{weight_norm:.4f}\t\t{weight_max:.4f}")
    
    # 找到最佳λ值（最小测试误差）
    best_idx = np.argmin(test_mses)
    best_lambda = lambdas[best_idx]
    print(f"\n最佳λ值: {best_lambda:.4e} (测试MSE: {test_mses[best_idx]:.4f})")
    
    # 可视化分析
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建2x2的子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. MSE vs λ曲线
    ax1.semilogx(lambdas[1:], train_mses[1:], 'b-o', markersize=4, label='训练MSE')
    ax1.semilogx(lambdas[1:], test_mses[1:], 'r-s', markersize=4, label='测试MSE')
    ax1.axvline(best_lambda, color='g', linestyle='--', alpha=0.7, label=f'最佳λ={best_lambda:.2e}')
    ax1.set_xlabel('正则化参数 λ')
    ax1.set_ylabel('均方误差 (MSE)')
    ax1.set_title('MSE vs 正则化参数')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 权重L2范数 vs λ
    ax2.semilogx(lambdas[1:], weight_norms[1:], 'g-^', markersize=4)
    ax2.set_xlabel('正则化参数 λ')
    ax2.set_ylabel('权重L2范数')
    ax2.set_title('权重大小 vs 正则化参数')
    ax2.grid(True, alpha=0.3)
    
    # 3. 权重系数随λ变化的轨迹
    all_weights = np.array(all_weights)
    feature_names = df.columns[:-1]
    
    # 只显示前8个特征的权重变化（避免图形过于复杂）
    for i in range(min(8, all_weights.shape[1])):
        ax3.semilogx(lambdas[1:], all_weights[1:, i], label=f'{feature_names[i][:10]}...', linewidth=1.5)
    ax3.set_xlabel('正则化参数 λ')
    ax3.set_ylabel('权重值')
    ax3.set_title('权重系数轨迹（前8个特征）')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. 偏差-方差权衡分析
    ax4.semilogx(lambdas[1:], np.array(test_mses[1:]) - np.array(train_mses[1:]), 'purple', marker='d', markersize=4)
    ax4.set_xlabel('正则化参数 λ')
    ax4.set_ylabel('测试MSE - 训练MSE')
    ax4.set_title('过拟合程度分析')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    fig_path = os.path.join(OUT_DIR, 'ridge_regularization_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n分析图表已保存至: {fig_path}")
    
    # 分析总结
    print(f"\n=== 正则化分析总结 ===")
    print(f"1. 无正则化 (λ=0): 训练MSE={train_mses[0]:.4f}, 测试MSE={test_mses[0]:.4f}")
    print(f"2. 最佳正则化 (λ={best_lambda:.2e}): 训练MSE={train_mses[best_idx]:.4f}, 测试MSE={test_mses[best_idx]:.4f}")
    print(f"3. 强正则化 (λ={lambdas[-1]:.2e}): 训练MSE={train_mses[-1]:.4f}, 测试MSE={test_mses[-1]:.4f}")
    print(f"4. 权重范数变化: {weight_norms[0]:.4f} -> {weight_norms[best_idx]:.4f} -> {weight_norms[-1]:.4f}")
    
    print(f"\n观察结论:")
    print(f"- 随着λ增大，权重逐渐收缩，模型复杂度降低")
    print(f"- 适当的正则化可以减少过拟合，提高泛化能力")
    print(f"- 过强的正则化会导致欠拟合，训练和测试误差都增大")


if __name__ == '__main__':
    main()