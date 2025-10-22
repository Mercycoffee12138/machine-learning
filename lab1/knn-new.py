import numpy as np
import random


import numpy as np
import random

def load_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :256]
    y = data[:, 256:]
    labels = np.argmax(y, axis=1)
    return X, labels

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def knn_predict(X_train, y_train, x_test, k):
    distances = [manhattan_distance(x_test, x) for x in X_train]
    idx = np.argsort(distances)[:k]
    votes = [y_train[i] for i in idx]
    counts = np.bincount(votes)
    max_vote = np.max(counts)
    candidates = np.where(counts == max_vote)[0]
    return random.choice(candidates)

def loo_evaluate(X, y, k):
    correct = 0
    n = len(X)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        x_test = X[i]
        y_test = y[i]
        y_pred = knn_predict(X_train, y_train, x_test, k)
        if y_pred == y_test:
            correct += 1
    accuracy = correct / n
    return accuracy

if __name__ == "__main__":
    # 1. 用训练集LOO筛选最优质数k
    prime_k = [1,3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    X_train, y_train = load_data("semeion_train.txt")
    X_test, y_test = load_data("semeion_test.txt")
    loo_results = []
    print("训练集LOO准确率：")
    for k in prime_k:
        acc = loo_evaluate(X_train, y_train, k)
        print(f"k={k}: {acc:.4f}")
        loo_results.append((k, acc))
    best_k, best_acc = max(loo_results, key=lambda x: x[1])
    print(f"\n最优k={best_k}, 训练集LOO准确率={best_acc:.4f}")

    # 2. 用最优k在测试集上测试
    print(f"\n用最优k={best_k}在测试集上测试：")
    correct = 0
    n_test = len(X_test)
    for i in range(n_test):
        x_test = X_test[i]
        y_true = y_test[i]
        y_pred = knn_predict(X_train, y_train, x_test, best_k)
        if y_pred == y_true:
            correct += 1
    test_acc = correct / n_test
    print(f"测试集准确率: {test_acc:.4f}")
