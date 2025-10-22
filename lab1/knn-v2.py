import numpy as np
import random

def load_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :256]
    y = data[:, 256:]
    labels = np.argmax(y, axis=1)
    return X, labels

def normalize(X):
    # z-score标准化
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x_test, k):
    distances = [euclidean_distance(x_test, x) for x in X_train]
    idx = np.argsort(distances)[:k]
    votes = {}
    for i in idx:
        label = y_train[i]
        # 距离越近权重越大
        weight = 1 / (distances[i] + 1e-8)
        votes[label] = votes.get(label, 0) + weight
    max_vote = max(votes.values())
    candidates = [label for label, v in votes.items() if v == max_vote]
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
    X, y = load_data("semeion.data")
    X = normalize(X)
    k = 5  # 可尝试1、3、5
    loo_evaluate(X, y, k)