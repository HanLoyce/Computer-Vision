# Fashion-MNIST 分类实验（NumPy 手写三层 MLP）

本项目使用 NumPy 从零实现多层感知机（MLP），完成 Fashion-MNIST 的训练、验证、测试与可视化分析，适合作为机器学习/模式识别课程实验。

## 项目简介

- 不依赖深度学习框架（如 PyTorch/TensorFlow），核心网络与反向传播均为手写实现。
- 支持训练集与验证集损失曲线、验证集准确率曲线。
- 支持混淆矩阵、第一层权重可视化和错分样本分析。
- 支持自动保存验证集最佳模型权重到 `best_model.pkl`。
- 数据文件缺失时可自动下载 `.gz` 版本 Fashion-MNIST 原始数据。

## 目录结构

- `main.py`：模块化入口脚本，执行超参数搜索、最终测试与可视化。
- `loading.py`：读取 IDX/.gz 数据，完成归一化与训练/验证划分。
- `model.py`：激活函数与 MLP 模型定义（前向、反向、参数更新、权重保存/加载）。
- `train.py`：训练循环、余弦退火学习率、验证评估与最佳模型保存。
- `visual.py`：学习曲线、混淆矩阵、权重图与错例图绘制。
- `complete_code.py`：单文件整合版（便于单脚本运行与展示）。
- `data/`：原始数据目录。

## 环境依赖

- Python 3.9+
- numpy
- matplotlib
- seaborn
- scikit-learn

安装方式：

```bash
pip install numpy matplotlib seaborn scikit-learn
```

## 数据准备

优先使用以下原始文件（位于 `data/` 目录）：

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

如果上述文件不存在，代码会尝试读取同名 `.gz` 文件；若仍不存在，则自动从官方地址下载 `.gz` 数据。

## 快速开始

在 `fashion_mnist_classifier` 目录执行：

```bash
python complete_code.py
```
或运行模块版：

```bash
python main.py
```




## 训练配置（当前代码默认）

- 输入维度：784（28x28 展平）
- 输出类别：10
- 候选隐藏层维度：`[64, 128]`
- 候选学习率：`[0.1, 0.01]`
- 候选 L2 正则：`[0.001, 0.01]`
- 训练轮数：500（每组超参数）
- 批大小：128
- 激活函数：ReLU
- 损失函数：交叉熵 + L2 正则
- 学习率策略：余弦退火（`min_lr=1e-4`）

## 运行输出

训练完成后会生成以下文件（默认保存在运行命令时的当前目录）：

- `best_model.pkl`：验证集最优参数。
- `learning_curves.png`：训练/验证损失与验证准确率曲线。
- `confusion_matrix.png`：测试集混淆矩阵。
- `weights_visualization.png`：第一层权重可视化。
- `error_analysis.png`：错分样本分析图。

控制台会输出：

- 每 100 轮训练日志（含学习率、训练损失、验证损失、验证准确率）。
- 网格搜索最佳参数组合。
- 最终测试集准确率。

## 课程实验对应关系

- 训练过程可视化：`learning_curves.png`
- 参数/权重可视化：`weights_visualization.png`
- 模型误差分析：`error_analysis.png` 与 `confusion_matrix.png`
- 可复现实验流程：模块化代码 + 单文件代码双版本

## 常见问题

1. 提示找不到数据文件

- 确认在项目目录下运行脚本。
- 检查 `data/` 路径是否存在且文件名正确。
- 网络受限时，自动下载可能失败，请手动放置数据文件。

2. 没有生成图片

- 图片保存到当前工作目录，不一定与脚本同目录。
- 训练未完成或中途中断时，部分图像不会生成。

3. 多次运行准确率略有波动

- 随机初始化和数据打乱会带来小幅波动，属正常现象。
