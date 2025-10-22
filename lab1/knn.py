import numpy as np
## import... 安装所需要的依赖库 

def loaddata(filename):
    data=np.loadtxt("semeion.data")
    x=data[:,:256]
    y=data[:,256:]
    labels=np.argmax(y,axis=1)
    return x,labels

def distance(a,b):
    return np.sqrt(np.sum(a,b)*np.sum(a,b))


def loo_eval(X, y, k):
    # 这部分为实验的核心代码
    return acc

# 主流程
raw = np.loadtxt('semeion.data.txt')
X, y = raw[:, :256], np.argmax(raw[:, 256:], 1)

for k in [1, 3, 5]:
    acc = loo_eval(X, y, k)
    print(f'k={k}  LOO 准确率 = {acc:.4f}')
