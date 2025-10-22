import numpy as np
from scipy.ndimage import rotate

def load_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :256]
    y = data[:, 256:]
    return X, y

def augment_rotate(img_flat):
    img = img_flat.reshape(16, 16)
    angle = np.random.choice([-10, -5, 5, 10])
    img_rot = rotate(img, angle, reshape=False, order=1, mode='nearest')
    return img_rot.flatten()

if __name__ == "__main__":
    X, y = load_data("semeion_train.txt")
    # 数据增强
    X_aug = np.array([augment_rotate(x) for x in X])
    # 拼接标签
    data_aug = np.hstack([X_aug, y])
    # 保存
    np.savetxt("semeion_train_augmented.txt", data_aug)
    print("增强数据已保存到 semeion_train_augmented.txt")