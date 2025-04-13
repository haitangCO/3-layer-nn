import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_weights_and_heatmap(npz_path, output_dir="vis_weights", channels=3):
    # 加载模型参数
    params = np.load(npz_path)
    if 'W1' not in params or 'W2' not in params:
        raise ValueError("参数文件中未找到 W1 或 W2")

    W1 = params['W1']  # 形状为 (input_dim, hidden_size)
    W2 = params['W2']  # 形状为 (hidden_size, num_classes)

    # CIFAR-10 输入是 32x32x3，所以 input_dim = 3072
    input_dim = 32 * 32 * channels
    if W1.shape[0] != input_dim:
        raise ValueError(f"W1 shape mismatch: expected ({input_dim}, ?), got {W1.shape}")

    hidden_size = W1.shape[1]
    print(f"[INFO] 可视化隐藏单元数: {hidden_size}")

    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)

    # W1 可视化：将每个隐藏单元的权重重塑为图像并保存
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()

    for i in range(min(16, hidden_size)):  # 只可视化前 16 个隐藏单元
        weight = W1[:, i]  # 取出第 i 个隐藏单元的权重，形状为 (3072,)
        img = weight.reshape(3, 32, 32).transpose(1, 2, 0)  # reshape 成 RGB 图片

        # 归一化到 0~1 用于显示
        img = (img - img.min()) / (img.max() - img.min())

        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Hidden Unit {i}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'W1_weights_visualization.png'))
    plt.close()

    # W2 可视化：热图显示隐藏层到输出的权重
    plt.figure(figsize=(10, 6))
    plt.imshow(W2, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Class Index')
    plt.ylabel('Hidden Unit Index')
    plt.title('W2 Heatmap (Hidden → Output)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'W2_heatmap.png'))
    plt.close()

    print(f"[DONE] 所有可视化图已保存到: {output_dir}/")

if __name__ == "__main__":
    visualize_weights_and_heatmap("best_model.npz")
