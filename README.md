# 3-layer-nn
# CIFAR-10 三层神经网络分类器

本项目实现了一个基于 Numpy 的三层全连接神经网络，用于对 CIFAR-10 图像数据集进行分类，具备训练、验证、超参数搜索和权重可视化等功能。

## 🧠 模型结构

- 输入层：3072（32x32x3）个神经元
- 隐藏层：ReLU 激活，支持超参数设置（如 256/512/1024）
- 输出层：10 个类别的 softmax 输出

## 📁 项目结构

```
├── main.py                  # 主程序入口，负责数据加载、训练、测试和可视化
├── three_layer_nn.py        # 三层神经网络类及训练/测试函数定义
├── hyper_search.py          # 网格搜索，寻找最优超参数
├── model_visualization.py   # 用于可视化训练后的权重
├── utils.py                 # 数据预处理与工具函数
├── grid_search_log.csv      # 网格搜索记录日志

```

## 🚀 快速开始

### 1. 安装依赖

确保你已安装 Python 3，并安装以下依赖：

```bash
pip install numpy matplotlib scikit-learn tqdm
```

### 2. 下载 CIFAR-10 数据集

请从官网：https://www.cs.toronto.edu/~kriz/cifar.html 下载并解压到当前目录的 `cifar-10-batches-py` 文件夹。

### 3. 运行主程序进行训练与测试

```bash
python main.py
```

默认会：
- 加载数据
- 运行网格搜索获取最佳超参数
- 进行训练与验证，保存最佳模型为 `best_model.npz`
- 输出测试集准确率
- 可视化 loss、accuracy 曲线和权重

### 4. 可视化训练权重

```bash
python model_visualization.py
```

会在vis_weights文件中生成可视化图像，其中：
- W1 表示输入到隐藏层的权重，反映提取的基础图像特征
- W2 表示隐藏层到输出层的权重，体现不同类别特征的组合

## 📊 最优超参数与测试结果

在 `grid_search_log.csv` 中记录了不同组合下的性能。最终选择了：

- 学习率：`0.01`
- 正则化强度：`0.001`
- 隐藏层大小：`512`
- Batch Size：`128`
- Epochs：`30`

最终在测试集上取得 **50%+ 准确率**（不同初始化下略有浮动）。



---

如需复现本实验，欢迎克隆仓库后参考上文步骤。欢迎 star 或提 issue 交流！
```
