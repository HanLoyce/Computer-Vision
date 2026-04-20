import numpy as np
import os 
import pickle
import gzip
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_URLS = {
    'train_images': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
    'train_labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
    'test_images': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
    'test_labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
}

FILENAME_TO_KEY = {
    'train-images-idx3-ubyte.gz': 'train_images',
    'train-labels-idx1-ubyte.gz': 'train_labels',
    't10k-images-idx3-ubyte.gz': 'test_images',
    't10k-labels-idx1-ubyte.gz': 'test_labels',
}

np.random.seed(123)
# ---------------加载数据-------------------
def load_data(data_dir='data'):
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(SCRIPT_DIR, data_dir)
    os.makedirs(data_dir, exist_ok=True)

    def ensure_dataset_file(filename):
        gz_filename = filename + '.gz'
        if os.path.exists(filename) or os.path.exists(gz_filename):
            return

        gz_basename = os.path.basename(gz_filename)
        if gz_basename not in FILENAME_TO_KEY:
            raise FileNotFoundError(f'No download URL configured for: {gz_basename}')

        url_key = FILENAME_TO_KEY[gz_basename]
        url = DATASET_URLS[url_key]
        print(f'Downloading {gz_basename} from {url} ...')
        urllib.request.urlretrieve(url, gz_filename)
        print(f'Saved to {gz_filename}')

    def read_images(filename):
        ensure_dataset_file(filename)
        if os.path.exists(filename + '.gz'):
            with gzip.open(filename + '.gz', 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
        else:
            with open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28*28) / 255.0  # 归一化到 0~1 之间
    
    def read_labels(filename):
        ensure_dataset_file(filename)
        if os.path.exists(filename + '.gz'):
            with gzip.open(filename + '.gz', 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
        else:
            with open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    
    X_train_full = read_images(os.path.join(data_dir,'train-images-idx3-ubyte'))
    y_train_full = read_labels(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    X_test = read_images(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    y_test = read_labels(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))

    X_train,y_train = X_train_full[:-10000],y_train_full[:-10000]
    X_val,y_val = X_train_full[-10000:],y_train_full[-10000:]

    return X_train,y_train,X_val,y_val,X_test,y_test

# ---------------搭建MLP-------------------
class ReLU:
    def forward(self,Z):
        self.Z = Z
        return np.maximum(0,Z)
    def backward(self,dA):
        return dA * (self.Z > 0).astype(float)
    
class Sigmoid:
    def forward(self,Z):
        self.A = 1 / (1 + np.exp(-np.clip(Z,-250,250)))
        return self.A
    def backward(self,dA):
        return dA * self.A * (1 - self.A)
    
class MLP:
    def __init__(self,input_dim=784,hidden_dim=128,output_dim=10,act='relu'):
        if act == 'relu':
            init_scale = np.sqrt(2./input_dim)
        else:
            init_scale = np.sqrt(1./input_dim)
        self.W1 = np.random.randn(input_dim,hidden_dim) * init_scale
        self.b1 = np.zeros((1,hidden_dim))

        init_scale2 = np.sqrt(2. / hidden_dim) if act == 'relu' else np.sqrt(1. / hidden_dim)
        self.W2 = np.random.randn(hidden_dim,output_dim) * init_scale2
        self.b2 = np.zeros((1,output_dim))

        self.activation = ReLU() if act == 'relu' else Sigmoid()

    def forward(self,X):
        self.X = X
        self.Z1 = np.dot(X,self.W1) + self.b1
        self.A1 = self.activation.forward(self.Z1)
        self.Z2 = np.dot(self.A1,self.W2) + self.b2

        exp_Z2 = np.exp(self.Z2 - np.max(self.Z2,axis=1,keepdims=True))
        self.A2 = exp_Z2 / np.sum(exp_Z2,axis=1,keepdims=True)
        return self.A2
    
    def compute_loss(self,y_true, l2_reg=0.0):
        m = y_true.shape[0]
        correct_logprobs = -np.log(self.A2[range(m),y_true] + 1e-15)
        data_loss = np.sum(correct_logprobs) / m
        reg_loss = 0.5 * l2_reg * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return data_loss + reg_loss
    
    def backward(self,y_true,l2_reg=0.0):
        m = y_true.shape[0]
        dZ2 = self.A2.copy()
        dZ2[range(m),y_true] -= 1
        dZ2 /= m

        #输出层梯度
        dW2 = np.dot(self.A1.T,dZ2) + l2_reg * self.W2
        db2 = np.sum(dZ2,axis = 0,keepdims=True)

        #隐藏层梯度
        dA1 = np.dot(dZ2,self.W2.T)
        dZ1 = self.activation.backward(dA1)
        dW1 = np.dot(self.X.T,dZ1) + l2_reg * self.W1
        db1 = np.sum(dZ1,axis=0,keepdims=True)

        return {'W1':dW1,'b1':db1,'W2':dW2,'b2':db2}
    
    def update(self,grads,lr):
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']

    def save_weights(self,filepath):
        if not os.path.isabs(filepath):
            filepath = os.path.join(SCRIPT_DIR, filepath)
        with open(filepath,'wb') as f:
            pickle.dump({'W1':self.W1,'b1':self.b1,'W2':self.W2,'b2':self.b2},f)
        
    def load_weights(self,filepath):
        if not os.path.isabs(filepath):
            filepath = os.path.join(SCRIPT_DIR, filepath)
        with open(filepath,'rb') as f:
            weights = pickle.load(f)
            self.W1,self.b1 = weights['W1'],weights['b1']
            self.W2,self.b2 = weights['W2'],weights['b2']


# ---------------训练循环与验证-------------------
def get_accuracy(y_pred,y_true):
    return np.mean(np.argmax(y_pred,axis=1) == y_true)

def train_model(model,X_train,y_train,X_val,y_val,epochs=50,batch_size=128,lr=0.1,l2_reg=0.001,min_lr=1e-4):
    history = {'train_loss':[],'val_loss':[],'val_acc':[]}
    best_val_acc = 0.0
    num_batches = X_train.shape[0] // batch_size
    base_lr = lr
    for epoch in range(epochs):
        if epochs > 1:
            current_lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * epoch / (epochs - 1)))
        else:
            current_lr = base_lr

        #打乱数据
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        for i in range(num_batches):
            X_batch = X_shuffled[i * batch_size : (i+1) * batch_size]
            y_batch = y_shuffled[i * batch_size : (i+1) * batch_size]

            #前向传播
            model.forward(X_batch)
            loss = model.compute_loss(y_batch,l2_reg)
            epoch_loss += loss

            #反向传播
            grads = model.backward(y_batch,l2_reg)
            model.update(grads,current_lr)
    
        val_probs = model.forward(X_val)
        val_loss = model.compute_loss(y_val,l2_reg)
        val_acc = get_accuracy(val_probs,y_val)

        train_loss_avg = epoch_loss / num_batches
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch+1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f} - Train Loss: {train_loss_avg:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    #保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights('best_model.pkl')
    return history

    
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
    fig, axes = plt.subplots(5, 6, figsize=(8, 8))
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



#-----------------主程序-------------------
FASHION_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
X_train, y_train, X_val, y_val, X_test, y_test = load_data('data')
learning_rates = [0.1,0.01]
hidden_dims = [64,128]
l2_regs = [0.001,0.01]

best_acc_overall = 0
best_params = {}
best_history = None

for lr in learning_rates:
    for hd in hidden_dims:
        for l2 in l2_regs:
            print(f"\n--- Training with LR: {lr:.4f}, HD: {hd}, L2: {l2:.4f} ---")
            model = MLP(hidden_dim = hd,act='relu')
            history = train_model(model,X_train,y_train,X_val,y_val,epochs=500,lr=lr,l2_reg=l2)
            val_acc = max(history['val_acc'])
            if val_acc > best_acc_overall:
                best_acc_overall = val_acc
                best_params = {'lr':lr,'hidden_dim':hd,'l2_reg':l2}
                best_history = history
print(f"\n[超参数查找完成] 最佳参数组合: {best_params}, 验证集最高准确率: {best_acc_overall:.4f}")


# 加载最佳模型并在测试集上评估
print("\n3. 加载最佳模型并在测试集上进行最终评估...")
# 用最佳隐藏层参数重新实例化网络，并加载刚才保存的权重
best_model = MLP(hidden_dim=best_params['hidden_dim'], act='relu')
best_model.load_weights('best_model.pkl')

# 测试集准确率
test_probs = best_model.forward(X_test)
test_preds = np.argmax(test_probs, axis=1)
test_acc = get_accuracy(test_probs, y_test)
print(f"独立测试集最终准确率 (Accuracy): {test_acc:.4f}")

print("\n4. 正在生成实验报告所需的各类图片...")
# 报告要求 1：可视化 Loss 曲线和 Accuracy 曲线
plot_learning_curves(best_history)

# 报告要求 2：权重可视化
visualize_weights(best_model)

# 报告要求 3：错例分析
error_analysis(best_model, X_test, y_test, FASHION_CLASSES)

# 基本要求：打印并保存混淆矩阵
plot_confusion_matrix(y_test, test_preds, FASHION_CLASSES)

print("\n全部任务执行完毕！图片已保存在当前目录下。")