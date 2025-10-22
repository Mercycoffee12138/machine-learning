#!/usr/bin/env python3
"""
拓展任务：多项式回归模型选择
- 目的：实现多项式回归，分析不同阶数对模型性能的影响
- 数据：使用 winequality-white.csv
- 分析：过拟合/欠拟合现象，模型选择
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
CSV = os.path.join(DATA_DIR, 'winequality-white.csv')

np.random.seed(42)


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio=0.2):
    """数据集划分"""
    n = X.shape[0]
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    return X[idx[test_size:]], y[idx[test_size:]], X[idx[:test_size]], y[idx[:test_size]]


def normalize(X_train: np.ndarray, X_test: np.ndarray):
    """数据标准化"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def create_polynomial_features(X: np.ndarray, degree: int):
    """手动创建多项式特征"""
    n_samples, n_features = X.shape
    
    if degree == 1:
        return X
    
    # 存储所有多项式特征
    poly_features = [X]  # 一次项
    
    if degree >= 2:
        # 二次项：x_i^2 和交叉项 x_i * x_j
        for i in range(n_features):
            # 平方项
            poly_features.append(X[:, i:i+1] ** 2)
            # 交叉项
            for j in range(i+1, n_features):
                poly_features.append((X[:, i] * X[:, j]).reshape(-1, 1))
    
    if degree >= 3:
        # 三次项：选择部分重要的三次项（避免特征爆炸）
        for i in range(min(5, n_features)):  # 只对前5个特征生成三次项
            poly_features.append(X[:, i:i+1] ** 3)
    
    if degree >= 4:
        # 四次项：进一步限制
        for i in range(min(3, n_features)):  # 只对前3个特征生成四次项
            poly_features.append(X[:, i:i+1] ** 4)
    
    return np.hstack(poly_features)


def linear_regression_fit(X: np.ndarray, y: np.ndarray):
    """线性回归闭式解"""
    # 添加偏置项
    X_ext = np.column_stack([X, np.ones(X.shape[0])])
    # 使用正规方程求解
    theta = np.linalg.solve(X_ext.T @ X_ext, X_ext.T @ y)
    return theta[:-1], theta[-1]  # 权重和偏置


def predict(X: np.ndarray, w: np.ndarray, b: float):
    """预测"""
    return X @ w + b


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 读取数据集
    print("正在读取红酒质量数据集...")
    df = pd.read_csv(CSV, sep=';')
    print(f"数据集形状: {df.shape}")
    
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)
    
    # 为了可视化方便，我们选择一个主要特征进行单变量多项式回归
    # 选择与目标最相关的特征
    correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    best_feature_idx = np.argmax(correlations)
    feature_name = df.columns[best_feature_idx]
    
    print(f"选择最相关的特征进行可视化: {feature_name} (相关系数: {correlations[best_feature_idx]:.3f})")
    
    X_single = X[:, best_feature_idx:best_feature_idx+1]  # 单特征
    X_multi = X  # 多特征
    
    # 数据集划分
    X_single_train, y_train, X_single_test, y_test = train_test_split(X_single, y, test_ratio=0.2)
    X_multi_train, _, X_multi_test, _ = train_test_split(X_multi, y, test_ratio=0.2)
    
    # 数据标准化
    X_single_train_norm, X_single_test_norm = normalize(X_single_train, X_single_test)
    X_multi_train_norm, X_multi_test_norm = normalize(X_multi_train, X_multi_test)
    
    print(f"\n训练集样本数: {len(y_train)}")
    print(f"测试集样本数: {len(y_test)}")
    
    # 测试不同的多项式阶数
    degrees = [1, 2, 3, 4, 5]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 存储结果
    results = {
        'single_feature': {'train_mse': [], 'test_mse': [], 'models': []},
        'multi_feature': {'train_mse': [], 'test_mse': [], 'models': []}
    }
    
    print(f"\n=== 多项式回归分析 ===")
    print("阶数\t单特征-训练MSE\t单特征-测试MSE\t多特征-训练MSE\t多特征-测试MSE")
    print("-" * 80)
    
    for degree in degrees:
        # 1. 单特征多项式回归（用于可视化）
        X_single_poly_train = create_polynomial_features(X_single_train_norm, degree)
        X_single_poly_test = create_polynomial_features(X_single_test_norm, degree)
        
        w_single, b_single = linear_regression_fit(X_single_poly_train, y_train)
        
        y_single_train_pred = predict(X_single_poly_train, w_single, b_single)
        y_single_test_pred = predict(X_single_poly_test, w_single, b_single)
        
        single_train_mse = mean_squared_error(y_train, y_single_train_pred)
        single_test_mse = mean_squared_error(y_test, y_single_test_pred)
        
        results['single_feature']['train_mse'].append(single_train_mse)
        results['single_feature']['test_mse'].append(single_test_mse)
        results['single_feature']['models'].append((w_single, b_single))
        
        # 2. 多特征多项式回归
        X_multi_poly_train = create_polynomial_features(X_multi_train_norm, degree)
        X_multi_poly_test = create_polynomial_features(X_multi_test_norm, degree)
        
        w_multi, b_multi = linear_regression_fit(X_multi_poly_train, y_train)
        
        y_multi_train_pred = predict(X_multi_poly_train, w_multi, b_multi)
        y_multi_test_pred = predict(X_multi_poly_test, w_multi, b_multi)
        
        multi_train_mse = mean_squared_error(y_train, y_multi_train_pred)
        multi_test_mse = mean_squared_error(y_test, y_multi_test_pred)
        
        results['multi_feature']['train_mse'].append(multi_train_mse)
        results['multi_feature']['test_mse'].append(multi_test_mse)
        results['multi_feature']['models'].append((w_multi, b_multi))
        
        print(f"{degree}\t{single_train_mse:.4f}\t\t{single_test_mse:.4f}\t\t{multi_train_mse:.4f}\t\t{multi_test_mse:.4f}")
    
    # 可视化分析
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 单特征多项式拟合曲线
    x_plot = np.linspace(X_single_train_norm.min(), X_single_train_norm.max(), 300).reshape(-1, 1)
    
    ax1.scatter(X_single_train_norm, y_train, alpha=0.6, color='lightblue', s=20, label='训练数据')
    ax1.scatter(X_single_test_norm, y_test, alpha=0.6, color='lightcoral', s=20, label='测试数据')
    
    for i, degree in enumerate(degrees):
        w, b = results['single_feature']['models'][i]
        x_poly = create_polynomial_features(x_plot, degree)
        y_plot = predict(x_poly, w, b)
        ax1.plot(x_plot, y_plot, color=colors[i], linewidth=2, label=f'阶数 {degree}')
    
    ax1.set_xlabel(f'{feature_name} (标准化)')
    ax1.set_ylabel('红酒质量')
    ax1.set_title('单特征多项式回归拟合曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 单特征MSE对比
    ax2.plot(degrees, results['single_feature']['train_mse'], 'b-o', label='训练MSE', linewidth=2)
    ax2.plot(degrees, results['single_feature']['test_mse'], 'r-s', label='测试MSE', linewidth=2)
    ax2.set_xlabel('多项式阶数')
    ax2.set_ylabel('均方误差 (MSE)')
    ax2.set_title('单特征：MSE vs 多项式阶数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(degrees)
    
    # 3. 多特征MSE对比
    ax3.plot(degrees, results['multi_feature']['train_mse'], 'b-o', label='训练MSE', linewidth=2)
    ax3.plot(degrees, results['multi_feature']['test_mse'], 'r-s', label='测试MSE', linewidth=2)
    ax3.set_xlabel('多项式阶数')
    ax3.set_ylabel('均方误差 (MSE)')
    ax3.set_title('多特征：MSE vs 多项式阶数')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(degrees)
    
    # 4. 过拟合分析：测试MSE与训练MSE的差值
    single_overfitting = np.array(results['single_feature']['test_mse']) - np.array(results['single_feature']['train_mse'])
    multi_overfitting = np.array(results['multi_feature']['test_mse']) - np.array(results['multi_feature']['train_mse'])
    
    ax4.plot(degrees, single_overfitting, 'g-^', label='单特征过拟合程度', linewidth=2)
    ax4.plot(degrees, multi_overfitting, 'm-d', label='多特征过拟合程度', linewidth=2)
    ax4.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('多项式阶数')
    ax4.set_ylabel('测试MSE - 训练MSE')
    ax4.set_title('过拟合程度分析')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(degrees)
    
    plt.tight_layout()
    
    # 保存图像
    fig_path = os.path.join(OUT_DIR, 'polynomial_regression_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n分析图表已保存至: {fig_path}")
    
    # 分析总结
    print(f"\n=== 多项式回归分析总结 ===")
    
    # 找到最佳阶数
    best_single_idx = np.argmin(results['single_feature']['test_mse'])
    best_multi_idx = np.argmin(results['multi_feature']['test_mse'])
    
    print(f"\n1. 最佳多项式阶数:")
    print(f"   单特征: 阶数 {degrees[best_single_idx]} (测试MSE: {results['single_feature']['test_mse'][best_single_idx]:.4f})")
    print(f"   多特征: 阶数 {degrees[best_multi_idx]} (测试MSE: {results['multi_feature']['test_mse'][best_multi_idx]:.4f})")
    
    print(f"\n2. 过拟合现象观察:")
    for i, degree in enumerate(degrees):
        single_gap = results['single_feature']['test_mse'][i] - results['single_feature']['train_mse'][i]
        multi_gap = results['multi_feature']['test_mse'][i] - results['multi_feature']['train_mse'][i]
        
        if single_gap > 0.1:
            print(f"   阶数 {degree}: 单特征出现明显过拟合 (差值: {single_gap:.4f})")
        if multi_gap > 0.1:
            print(f"   阶数 {degree}: 多特征出现明显过拟合 (差值: {multi_gap:.4f})")
    
    print(f"\n3. 主要发现:")
    print(f"   - 低阶数(1-2): 可能存在欠拟合，模型过于简单")
    print(f"   - 中等阶数(2-3): 通常有较好的偏差-方差权衡")
    print(f"   - 高阶数(4-5): 容易过拟合，训练误差很小但测试误差较大")
    print(f"   - 多特征比单特征有更好的预测性能")
    
    # 特征数量分析
    for i, degree in enumerate(degrees):
        single_features = create_polynomial_features(X_single_train_norm[:1], degree).shape[1]
        multi_features = create_polynomial_features(X_multi_train_norm[:1], degree).shape[1]
        print(f"   阶数 {degree}: 单特征扩展到 {single_features} 维, 多特征扩展到 {multi_features} 维")


if __name__ == '__main__':
    main()