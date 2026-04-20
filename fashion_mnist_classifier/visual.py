import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------可视化与测试分析-------------------、
# 1. 绘制 Loss 和 Accuracy 曲线 
def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig('learning_curves.png')
    plt.show()

# 2. 混淆矩阵 
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

# 3. 权重可视化与空间模式观察 
def visualize_weights(model):
    W1 = model.W1 # 形状为 (784, hidden_dim)
    # 取前 16 个隐藏神经元的权重可视化
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        weight_img = W1[:, i].reshape(28, 28)
        ax.imshow(weight_img, cmap='seismic') # seismic红蓝分明，代表正负权重
        ax.axis('off')
    plt.suptitle('Visualization of First Layer Weights')
    plt.savefig('weights_visualization.png')
    plt.show()

# 4. 错例分析 (Error Analysis) 
def error_analysis(model, X_test, y_test, classes):
    probs = model.forward(X_test)
    y_pred = np.argmax(probs, axis=1)
    errors = np.where(y_pred != y_test)[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for i, ax in enumerate(axes.flat):
        idx = errors[i]
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {classes[y_test[idx]]}\nPred: {classes[y_pred[idx]]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('error_analysis.png')
    plt.show()
