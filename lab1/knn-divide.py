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

def evaluate(X_train, y_train, X_test, y_test, k):
    correct = 0
    n = len(X_test)
    for i in range(n):
        x_test = X_test[i]
        y_true = y_test[i]
        y_pred = knn_predict(X_train, y_train, x_test, k)
        if y_pred == y_true:
            correct += 1
        if (i+1) % 100 == 0:
            print(f"已完成 {i+1}/{n}")
    accuracy = correct / n
    print(f"测试准确率: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    # 加载训练数据和测试数据
    X_train, y_train = load_data("semeion_train.txt")
    X_test, y_test = load_data("semeion_test.txt")
    
    print(f"训练数据: {X_train.shape[0]} 样本")
    print(f"测试数据: {X_test.shape[0]} 样本")
    
    k = 17
    evaluate(X_train, y_train, X_test, y_test, k)