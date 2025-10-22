import numpy as np
import random

def load_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :256]
    y = data[:, 256:]
    labels = np.argmax(y, axis=1)
    return X, labels

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x_test, k):
    distances = [euclidean_distance(x_test, x) for x in X_train]
    idx = np.argsort(distances)[:k]
    votes = [y_train[i] for i in idx]
    # 投票
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
        if (i+1) % 100 == 0:
            print(f"已完成 {i+1}/{n}")
    accuracy = correct / n
    print(f"LOO准确率: {accuracy:.4f}")

if __name__ == "__main__":
    # 加载训练集和测试集
    X_train, y_train = load_data("semeion_train.txt")
    X_test, y_test = load_data("semeion_test.txt")
    k_list = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    results = []
    n_test = len(X_test)
    for k in k_list:
        print(f"\n测试k={k}")
        print("训练集LOO准确率:")
        loo_evaluate(X_train, y_train, k)
        correct = 0
        for i in range(n_test):
            x_test = X_test[i]
            y_true = y_test[i]
            y_pred = knn_predict(X_train, y_train, x_test, k)
            if y_pred == y_true:
                correct += 1
        accuracy = correct / n_test
        print(f"测试集准确率: {accuracy:.4f}")
        results.append((k, accuracy))

    print("\n所有k值测试集准确率汇总：")
    for k, acc in results:
        print(f"k={k}: {acc:.4f}")
    best_k, best_acc = max(results, key=lambda x: x[1])
    print(f"\n最优k={best_k}, 测试集准确率={best_acc:.4f}")