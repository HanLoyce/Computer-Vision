# Fashion-MNIST MLP Classifier

基于 NumPy 从零实现的 Fashion-MNIST 图像分类项目，包含数据加载、模型训练、测试评估、可视化分析与错例分析完整流程。

## 1. 项目亮点

- 纯 NumPy 手写两层 MLP（不依赖深度学习框架）
- 支持训练集与验证集 Loss 曲线、验证集 Accuracy 曲线可视化
- 支持第一层权重可视化（空间模式观察）
- 支持测试集错例分析与混淆矩阵
- 保存验证集最优模型权重到 best_model.pkl

## 2. 项目结构

- [main.py](main.py): 模块化主入口，训练 + 评估 + 可视化
- [loading.py](loading.py): Fashion-MNIST 原始 IDX 数据读取与预处理
- [model.py](model.py): 激活函数与 MLP 模型定义
- [train.py](train.py): 训练循环、学习率调度、验证与模型保存
- [visual.py](visual.py): 学习曲线、混淆矩阵、权重图、错例图
- [complete_code.py](complete_code.py): 单文件整合版本（含超参数搜索）
- [best_model.pkl](best_model.pkl): 已训练的最优权重
- [data](data): 原始数据文件目录
- [项目报告.md](项目报告.md): 实验报告 Markdown 版本

## 3. 环境依赖

推荐 Python 3.9 及以上。

安装依赖：

```bash
pip install numpy matplotlib seaborn scikit-learn
```

## 4. 数据准备

请确保以下文件位于 [data](data) 目录下：

- train-images-idx3-ubyte
- train-labels-idx1-ubyte
- t10k-images-idx3-ubyte
- t10k-labels-idx1-ubyte

说明：代码同时支持读取对应 .gz 压缩文件。

## 5. 运行方式

在项目目录执行：

```bash
python main.py
```

可选：运行单文件版本（含网格搜索逻辑）：

```bash
python complete_code.py
```

## 6. 训练与评估说明

- 输入维度: 784（28x28 展平）
- 隐藏层维度: 128（默认）
- 输出维度: 10
- 激活函数: ReLU
- 损失函数: 交叉熵 + L2 正则
- 优化方式: 小批量梯度下降
- 学习率策略: 余弦退火

当前已保存权重在测试集上的准确率为：

- Test Accuracy: 0.8372

## 7. 输出文件说明

运行完成后会生成以下结果：

- learning_curves.png: 训练/验证 Loss 与验证 Accuracy 曲线
- confusion_matrix.png: 混淆矩阵
- weights_visualization.png: 第一层权重可视化
- error_analysis.png: 错例分析可视化
- best_model.pkl: 验证集最优权重

提示：图像保存路径取决于运行命令时的当前工作目录。

## 8. 与课程作业要求对应

- 要求 1（训练过程可视化）: 已实现 learning_curves.png
- 要求 2（权重可视化与空间模式观察）: 已实现 weights_visualization.png
- 要求 3（错例分析）: 已实现 error_analysis.png
- 要求 4（代码提交说明）: 本 README 已提供依赖与运行方式说明

## 9. 提交信息占位（提交前请补全）

- GitHub Repo 链接: https://github.com/<your-username>/<your-repo>
- 模型权重下载地址: https://...
- 报告最后更新时间: YYYY-MM-DD HH:mm

## 10. 常见问题

1. 运行时报找不到数据文件
- 检查 [data](data) 目录与文件名是否正确。
- 确认是在项目根目录执行 python main.py。

2. 没有看到可视化图片
- 检查命令执行目录，图片会保存到当前工作目录。

3. 准确率波动
- 该实现默认未固定随机种子，不同运行可能有小幅波动。
