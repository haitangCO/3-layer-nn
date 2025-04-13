# main.py
import numpy as np
import matplotlib.pyplot as plt
from three_layer_nn import ThreeLayerNN, train, test
from utils import load_cifar10_data, normalize_data, one_hot_encode, train_val_split, plot_training_curves
from hyper_search import grid_search

if __name__ == '__main__':
    # Load CIFAR-10 dataset
    X_train, y_train, X_test, y_test = load_cifar10_data()
    X_train, X_test = normalize_data(X_train), normalize_data(X_test)

    # Optional: Perform hyperparameter search
    print("\n[Optional] Running grid search on training set...")
    best_config = grid_search(X_train, y_train, use_small_batch=False)
    print("Best config:", best_config)

    # Use best config from grid search (or manually set)
    model = ThreeLayerNN(
        input_size=32*32*3,
        hidden_size=best_config['hidden_size'],
        output_size=10,
        activation='relu',
        weight_scale=1e-2
    )

    # Split validation set
    X_train, X_val, y_train,  y_val = train_val_split(X_train, y_train, val_ratio=0.1)

    # Train model with tracking
    history = train(
        model, X_train, y_train, X_val, y_val,
        epochs=30,
        batch_size=128,
        learning_rate=best_config['learning_rate'],
        lr_decay=0.9,
        reg=best_config['reg'],
        save_path='best_model.npz'
    )

    # Plot training curves
    plot_training_curves(history)

    # Evaluate on test set
    model.load('best_model.npz')
    test(model, X_test, y_test)

