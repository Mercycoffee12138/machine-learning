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
        X, y = load_data("semeion_augmented.data")
        k_list = [1, 3, 5, 7,  11, 13, 17]
        results = []
        n = len(X)
        indices = np.arange(n)
        np.random.seed(42)  # 保证每次分折一致
        np.random.shuffle(indices)
        folds = np.array_split(indices, 10)
        for k in k_list:
            print(f"\n测试k={k}")
            acc_list = []
            for fold in folds:
                test_idx = fold
                train_idx = np.setdiff1d(indices, test_idx)
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                correct = 0
                for i in range(len(X_test)):
                    x_test = X_test[i]
                    y_true = y_test[i]
                    y_pred = knn_predict(X_train, y_train, x_test, k)
                    if y_pred == y_true:
                        correct += 1
                acc = correct / len(X_test)
                acc_list.append(acc)
            mean_acc = np.mean(acc_list)
            print(f"10折交叉验证平均准确率: {mean_acc:.4f}")
            results.append((k, mean_acc))

        print("\n所有k值10折交叉验证准确率汇总：")
        for k, acc in results:
            print(f"k={k}: {acc:.4f}")
        best_k, best_acc = max(results, key=lambda x: x[1])
        print(f"\n最优k={best_k}, 10折交叉验证准确率={best_acc:.4f}")