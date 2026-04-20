import numpy as np
import os 
from loading import load_data
from visual import plot_learning_curves,plot_confusion_matrix
from visual import visualize_weights,error_analysis
from train import train_model
from train import get_accuracy
from model import MLP



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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