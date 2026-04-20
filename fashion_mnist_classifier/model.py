import numpy as np
import os 
import pickle
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
