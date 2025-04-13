
# utils.py
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_cifar10_data(data_dir='cifar-10-batches-py'):
    def load_batch(filename):
        with open(filename, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            X = dict[b'data'].astype('float32') / 255.0
            y = np.array(dict[b'labels'])
        return X, y

    xs, ys = [], []
    for i in range(1, 6):
        f = os.path.join(data_dir, f'data_batch_{i}')
        X, y = load_batch(f)
        xs.append(X)
        ys.append(y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    X_test, y_test = load_batch(os.path.join(data_dir, 'test_batch'))
    return X_train, y_train, X_test, y_test

def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-7
    return (X - mean) / std

def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def train_val_split(X, y, val_ratio=0.1):
    return train_test_split(X, y, test_size=val_ratio, random_state=42)

def plot_training_curves(history):
    epochs = np.arange(len(history['train_loss']))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
