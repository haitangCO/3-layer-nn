# hyper_search.py
# hyper_search.py

import numpy as np
import csv
import time
from three_layer_nn import ThreeLayerNN, train
from utils import train_val_split
from tqdm import tqdm

def grid_search(X_full, y_full, val_ratio=0.1, use_small_batch=True, log_csv='grid_search_log.csv', early_stop_delta=0.0001):
    learning_rates = [1e-1, 1e-2, 1e-3]
    regs = [0.1, 1e-2, 1e-3]
    hidden_sizes = [256, 512, 1024]

    best_val_acc = 0
    best_config = None
    no_improve_count = 0

    
    if use_small_batch:
        sample_limit = 5000
        val_count = int(sample_limit * val_ratio)
        train_count = sample_limit - val_count
        print(f"[DEBUG] Using only {sample_limit} samples for quick testing: {train_count} train + {val_count} val")
        X_full = X_full[:sample_limit]
        y_full = y_full[:sample_limit]

    total_runs = len(learning_rates) * len(regs) * len(hidden_sizes)
    search_iter = tqdm(total=total_runs, desc="Grid Search")

    # Open CSV file for logging results
    with open(log_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['learning_rate', 'reg', 'hidden_size', 'val_accuracy', 'elapsed_time_sec'])

        for lr in learning_rates:
            for reg in regs:
                for hidden in hidden_sizes:
                    print(f"\n--- Training with lr={lr}, reg={reg}, hidden={hidden} ---")
                    start_time = time.time()

                    model = ThreeLayerNN(
                        input_size=32*32*3,
                        hidden_size=hidden,
                        output_size=10,
                        activation='relu',
                        weight_scale=1e-2
                    )

                    # Re-split for each experiment to ensure correct batch sizing
                    X_train, X_val, y_train, y_val = train_val_split(X_full, y_full, val_ratio=val_ratio)
                   
                    N_train = X_train.shape[0]
                    batch_size = 64 if use_small_batch else 128
                    if batch_size > N_train:
                        batch_size = N_train

                    history = train(
                        model, X_train, y_train, X_val, y_val,
                        epochs=10,
                        batch_size=batch_size,
                        learning_rate=lr,
                        lr_decay=0.95,
                        reg=reg,
                        save_path="best_model.npz"
                    )

                    val_acc = history['val_acc'][-1]
                    elapsed = time.time() - start_time
                    print(f"Final Validation Accuracy: {val_acc:.4f} | Time: {elapsed:.2f} sec")

                    writer.writerow([lr, reg, hidden, val_acc, elapsed])

                    if val_acc > best_val_acc + early_stop_delta:
                        best_val_acc = val_acc
                        best_config = {
                            'learning_rate': lr,
                            'reg': reg,
                            'hidden_size': hidden
                        }
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    print("Current Best Config:", best_config)
                    print(f"Best Validation Accuracy So Far: {best_val_acc:.4f}")

                    if no_improve_count >= 10:
                        print("[EARLY STOP] No improvement for 5 configurations.")
                        search_iter.update(total_runs - search_iter.n)
                        search_iter.close()
                        return best_config

                    search_iter.update(1)

    search_iter.close()
    print("\n=== Best Configuration Found ===")
    print(best_config)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    return best_config
