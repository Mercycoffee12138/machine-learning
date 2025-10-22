import numpy as np
from scipy.ndimage import rotate
import random

def augment_image(image, angle_range=(-10, 10)):
    """对单张图像进行旋转增强
    
    Args:
        image: 256维的一维向量，表示16x16的图像
        angle_range: 旋转角度范围，默认(-10, 10)度
    
    Returns:
        增强后的图像，256维一维向量
    """
    angle = random.uniform(angle_range[0], angle_range[1])
    rotated = rotate(image.reshape(16, 16), angle, reshape=False, order=1)  # 双线性插值
    return rotated.flatten()

def augment_dataset(X, y, num_augmented_per_image=2):
    """对数据集进行增强
    
    Args:
        X: 图像数据，形状为(n_samples, 256)
        y: 标签数据，可以是one-hot编码或类别标签
        num_augmented_per_image: 每张图像生成的增强图像数量
    
    Returns:
        增强后的图像数据和标签数据
    """
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        # 原图
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        
        # 增强图像
        for _ in range(num_augmented_per_image):
            aug_img = augment_image(X[i], angle_range=(-10, 10))
            X_augmented.append(aug_img)
            y_augmented.append(y[i])
    
    return np.array(X_augmented), np.array(y_augmented)

def load_semeion_data(filename):
    """加载semeion.data文件
    
    Args:
        filename: semeion.data文件路径
    
    Returns:
        X: 图像数据，形状为(n_samples, 256)
        y_onehot: one-hot编码的标签，形状为(n_samples, 10)
        y_labels: 类别标签，形状为(n_samples,)
    """
    data = np.loadtxt(filename)
    X = data[:, :256]  # 前256列是图像数据
    y_onehot = data[:, 256:]  # 后10列是one-hot标签
    y_labels = np.argmax(y_onehot, axis=1)  # 转换为类别标签
    return X, y_onehot, y_labels

def augment_semeion_dataset(filename, num_augmented_per_image=2, output_file=None, keep_onehot=True):
    """直接处理semeion.data文件进行数据增强
    
    Args:
        filename: 输入的semeion.data文件路径
        num_augmented_per_image: 每张图像生成的增强图像数量
        output_file: 输出文件路径，如果为None则不保存文件
        keep_onehot: 是否保持one-hot编码格式，True则保持，False则转为类别标签
    
    Returns:
        X_aug: 增强后的图像数据
        y_aug: 增强后的标签数据
    """
    print(f"Loading data file: {filename}")
    X, y_onehot, y_labels = load_semeion_data(filename)
    print(f"Original dataset size: {X.shape[0]} samples")
    
    # Choose label format based on keep_onehot parameter
    y_input = y_onehot if keep_onehot else y_labels
    
    print("Performing data augmentation...")
    X_aug, y_aug = augment_dataset(X, y_input, num_augmented_per_image)
    print(f"Augmented dataset size: {X_aug.shape[0]} samples")
    
    # Save to file if needed
    if output_file:
        print(f"Saving augmented data to: {output_file}")
        if keep_onehot:
            # Keep semeion.data original format
            augmented_data = np.hstack([X_aug, y_aug])
        else:
            # Convert class labels back to one-hot encoding
            y_onehot_aug = np.eye(10)[y_aug.astype(int)]
            augmented_data = np.hstack([X_aug, y_onehot_aug])
        
        np.savetxt(output_file, augmented_data, fmt='%.4f')
        print("Data saved successfully!")
    
    return X_aug, y_aug

def visualize_augmentation(original_image, augmented_images, title="Data Augmentation Effect"):
    """Visualize data augmentation effect (optional function)
    
    Args:
        original_image: Original image, 256-dim vector
        augmented_images: List of augmented images
        title: Chart title
    """
    try:
        import matplotlib.pyplot as plt
        
        num_images = len(augmented_images) + 1
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
        
        # Display original image
        axes[0].imshow(original_image.reshape(16, 16), cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Display augmented images
        for i, aug_img in enumerate(augmented_images):
            axes[i + 1].imshow(aug_img.reshape(16, 16), cmap='gray')
            axes[i + 1].set_title(f'Augmented Image {i + 1}')
            axes[i + 1].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not installed, cannot visualize")

# 使用示例
if __name__ == "__main__":
    # Basic usage: augment semeion.data and save
    X_aug, y_aug = augment_semeion_dataset(
        filename="semeion.data",
        num_augmented_per_image=2,
        output_file="semeion_augmented.data",
        keep_onehot=True
    )
    
    # Visualization example (if matplotlib is installed)
    original_img = X_aug[0]  # First original image
    aug_imgs = [X_aug[1], X_aug[2]]  # Corresponding augmented images
    visualize_augmentation(original_img, aug_imgs)