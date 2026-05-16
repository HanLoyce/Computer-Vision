# U-Net 在 Oxford-IIIT Pet 上的语义分割实验报告

## 项目概述
本项目使用 Oxford-IIIT Pet 数据集完成语义分割实验，基于 U-Net 比较 Cross-Entropy、Dice 和组合损失三种训练方案的效果。报告重点说明数据划分方式、训练设置、mIoU 结果以及不同损失函数对分割性能的影响。

## 1. 模型结构、数据集与结果简介

### 1.1 模型结构
- 模型: U-Net（编码器-解码器结构，含跳跃连接）
- 输入: 3 通道 RGB 图像
- 输出: 3 类分割（前景/背景/边界，trimap 映射后标签为 0/1/2）

### 1.2 数据集
- 数据集: Oxford-IIIT Pet segmentation（trimaps）
- 官方划分: trainval=3680, test=3669
- 本实验训练策略:
  - 从 trainval 中每个 epoch 随机按 0.2 切分验证集
  - train=2944, val=736（每轮随机重采样）
  - test 保持独立，不参与训练/验证

### 1.3 主要结果
- Cross-Entropy: best val mIoU=0.824683（epoch 50）
- Dice: best val mIoU=0.801269（epoch 49）
- CE + Dice: best val mIoU=0.826622（epoch 49）
- 最优损失组合: CE + Dice

## 2. 详细实验设置

### 2.1 数据划分与迭代规模
- 每轮划分: train=2944, val=736（来自 trainval）
- batch size: 8
- 每个 epoch 的 iteration（向上取整）:
  - train iterations/epoch = ceil(2944/8) = 368
  - val iterations/epoch = ceil(736/8) = 92

### 2.2 网络与训练超参数
- 网络: U-Net（base channels=32）
- 输入尺寸: 256x256
- epoch: 50（当前结果汇总对应训练）
- 优化器: Adam
- 学习率: 1e-3
- 数据增强: 训练阶段随机水平翻转
- 输出目录: `Unet/results`

### 2.3 loss function 与评价指标
- loss function:
  - Cross-Entropy
  - Dice
  - CE + Dice
- 评价指标:
  - train/val loss
  - train/val mIoU（语义分割主指标）

说明：语义分割任务通常使用 mIoU 作为 Accuracy/mAP 的对应质量指标。

## 3. 实验结果

| loss | best val mIoU | best epoch |
|---|---:|---:|
| Cross-Entropy | 0.824683 | 50 |
| Dice | 0.801269 | 49 |
| CE + Dice | 0.826622 | 49 |

结论：在当前设置下，CE + Dice 的验证集 mIoU 最优。

## 4. wandb/swanlab 可视化截图


### 4.1 训练集 Loss 曲线（wandb）
![CE Curves](wandb/run-20260512_212516-knnswopg/files/media/images/ce_training_curves_151_c872918857ba6929323c.png)

### 4.2 训练集 mIoU 曲线（wandb）
![Combined Curves](wandb/run-20260512_212516-knnswopg/files/media/images/combined_training_curves_153_08ce5e827e27dad15596.png)

### 4.3 补充训练曲线截图
![Dice Curves](wandb/run-20260512_212516-knnswopg/files/media/images/dice_training_curves_152_2e0436986fbcb05b4524.png)

### 4.4 训练曲线与准确率对比分析

训练损失曲线显示，三种损失函数在训练过程中均能稳定下降并逐步收敛：Cross-Entropy 的训练 loss 下降较为平滑，Dice 在训练早期收敛更快但后期增益减小，CE + Dice 在训练中后期表现出既有收敛稳定性又有一定的早期加速特性。训练阶段的 mIoU 曲线也表明三者在前期快速提升，随后进入相对平稳的增长区间。

将训练曲线与独立测试集结果（见 4.5）结合判断：训练曲线表明模型已充分学习到主要语义特征且无明显过拟合迹象，三种损失在训练过程中的差异与测试集上小幅度的 mIoU 差别（最大 < 0.03）是一致的——即模型训练稳定但不同损失带来的微小性能差异并不显著。基于训练曲线与测试集对比，推荐在报告中以测试集 mIoU 为最终评价并直接展示训练曲线以说明训练稳定性。

### 4.5 测试集 mIoU 分析

我们在官方独立测试集上对三种已保存的最佳模型（results/*/best.pt）进行了评估，得到的 test mIoU 如下：

- Dice = 0.7760
- CE + Dice = 0.7689
- Cross-Entropy = 0.7620

详见 `Unet/results/test_summary.csv` 与 `Unet/results/test_miou_comparison.png`。

从测试集结果看，Dice 在本次测试集上取得最高的 mIoU，但三者差距较小（最大差异 < 0.03），表明模型在该任务上的基线性能较为稳定。建议以测试集 mIoU 作为最终评估指标，并在报告中直接展示测试集的对比图与数值表以便读者判断。

## 5. 结论
- U-Net 在 Oxford-IIIT Pet 分割任务上可稳定收敛，测试集 mIoU 达到约 0.77–0.78 的水平。
- 在独立测试集上，Dice 表现最好（mIoU=0.7760），但三种损失函数的差异较小（最大差异 < 0.03），说明模型基线稳定。
- 建议在报告中以测试集 mIoU 为最终评估指标，并同时提供测试集的对比图与数值表以便阅读者判断。

## 6. 结果文件来源
- `Unet/results/summary.csv`
- `Unet/results/summary.json`
- `Unet/results/*/history.json`
