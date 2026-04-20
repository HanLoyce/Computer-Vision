import numpy as np
import os 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------训练循环与验证-------------------
def get_accuracy(y_pred,y_true):
    return np.mean(np.argmax(y_pred,axis=1) == y_true)

def train_model(model,X_train,y_train,X_val,y_val,epochs=50,batch_size=128,lr=0.1,lr_decay=0.99,l2_reg=0.001,min_lr=1e-4):
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