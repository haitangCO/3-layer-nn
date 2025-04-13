# three_layer_nn.py
import numpy as np
import os
from typing import Tuple

# CIFAR-10 utility functions for loading and preprocessing
from utils import load_cifar10_data, normalize_data, one_hot_encode, train_val_split


class ThreeLayerNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, activation: str = 'relu', weight_scale=1e-3):
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * weight_scale,
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, output_size) * weight_scale,
            'b2': np.zeros(output_size),
        }
        self.activation = activation

    def _activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation")

    def _activation_grad(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2

    def forward(self, X):
        z1 = X.dot(self.params['W1']) + self.params['b1']
        a1 = self._activation(z1)
        scores = a1.dot(self.params['W2']) + self.params['b2']
        cache = (X, z1, a1)
        return scores, cache

    def loss(self, scores, y, reg):
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        N = y.shape[0]
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.mean(correct_logprobs)
        reg_loss = 0.5 * reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        loss = data_loss + reg_loss
        return loss, probs

    def backward(self, probs, y, cache, reg):
        grads = {}
        X, z1, a1 = cache
        N = X.shape[0]

        dscores = probs
        dscores[range(N), y] -= 1
        dscores /= N

        grads['W2'] = a1.T.dot(dscores) + reg * self.params['W2']
        grads['b2'] = np.sum(dscores, axis=0)

        da1 = dscores.dot(self.params['W2'].T)
        dz1 = da1 * self._activation_grad(z1)

        grads['W1'] = X.T.dot(dz1) + reg * self.params['W1']
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    def update_params(self, grads, learning_rate):
        for param in self.params:
            self.params[param] -= learning_rate * grads[param]

    def save(self, path):
        """Save the model parameters as a .npz file."""
        np.savez(path, **self.params)  # Save parameters to .npz format
        print(f"Model parameters saved to {path}")

    def load(self, path):
        """Load the model parameters from a .npz file."""
        loaded_params = np.load(path)
        self.params = {k: v for k, v in loaded_params.items()}  # Load parameters
        print(f"Model parameters loaded from {path}")

    def predict(self, X):
        scores, _ = self.forward(X)
        return np.argmax(scores, axis=1)


# Training loop

def train(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=128,
          learning_rate=0.1, lr_decay=0.95, reg=1e-3, save_path='best_model.npz',
          early_stopping=True, patience=3):

    N = X_train.shape[0]
    best_val_acc = 0.0
    best_params = None
    wait = 0  # 没提升的 epoch 次数

    # 初始化历史记录
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Shuffle data
        indices = np.arange(N)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0
        num_batches = 0

        for i in range(0, N, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            scores, cache = model.forward(X_batch)
            loss, probs = model.loss(scores, y_batch, reg)
            grads = model.backward(probs, y_batch, cache, reg)
            model.update_params(grads, learning_rate)

            epoch_loss += loss
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_loss_history.append(avg_train_loss)

        # 验证集损失和准确率
        val_scores, _ = model.forward(X_val)
        val_loss, _ = model.loss(val_scores, y_val, reg)
        val_loss_history.append(val_loss)

        val_pred = model.predict(X_val)
        val_acc = np.mean(val_pred == y_val)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1}: val_acc = {val_acc:.4f}, learning_rate = {learning_rate:.5f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}
            wait = 0
            if save_path:
                np.savez(save_path, **model.params)
        else:
            wait += 1
            if early_stopping and wait >= patience:
                print(f"[Early Stopping] No improvement for {patience} epochs. Stopping at epoch {epoch+1}.")
                break

        # 学习率衰减
        learning_rate *= lr_decay

    # 加载最佳参数
    if best_params:
        model.params = best_params

    # 返回训练记录
    return {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history
    }




def test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {acc:.4f}")
    return acc
