# 期中项目总 README

这个工作区包含三个彼此独立、但主题一致的计算机视觉项目：图像分类、语义分割和目标检测/多目标跟踪。这里把三个子项目的说明合并成一个总 README，方便直接按项目查阅和运行。

## 总览

| 目录 | 任务 | 核心入口 |
| --- | --- | --- |
| [ImageNet/](ImageNet/) | Oxford-IIIT Pet 迁移学习、超参数对比、预训练消融、SE 注意力对比 | [complete_code.py](ImageNet/complete_code.py) |
| [Unet/](Unet/) | Oxford-IIIT Pet 语义分割 | [train.py](Unet/train.py)、[eval_test.py](Unet/eval_test.py) |
| [YOLO/](YOLO/) | VisDrone 检测与多目标跟踪 | [scripts/run_train.py](YOLO/scripts/run_train.py)、[scripts/run_test_tracking.py](YOLO/scripts/run_test_tracking.py) |

## 环境依赖

三个项目可以分别装依赖，也可以按需要统一安装。

- ImageNet 项目：`torch`、`torchvision`、`matplotlib`、`pandas`、`numpy`，可选 `wandb`
- U-Net 项目：`torch`、`numpy`、`pillow`、`matplotlib`，可选 `wandb`
- YOLO 项目：见 [YOLO/requirements.txt](YOLO/requirements.txt)

如果你只想先跑某一个项目，建议直接进入对应目录后再安装依赖。

## 项目一：ImageNet 目录的分类实验

### 主要内容

- 使用 ResNet18 / ResNet34 做迁移学习微调
- 对不同 epochs 和学习率组合做对比
- 对比预训练和随机初始化
- 引入 SEBlock 做注意力对比

### 依赖安装

```bash
pip install torch torchvision matplotlib pandas numpy wandb
```

### 数据准备

脚本默认会优先查找 `../data/oxford-iiit-pet`，找不到时会自动下载数据集。

也可以手动指定数据目录。

### 运行方式

```bash
cd ImageNet
python complete_code.py
```

启用 W&B：

```bash
python complete_code.py --use-wandb
```

常用参数：

- `--data-root`：数据集目录
- `--output`：结果 JSON 保存路径
- `--use-wandb`：启用 W&B 记录
- `--use-swanlab`：兼容旧参数名，效果同 `--use-wandb`

### 输出结果

默认输出到 [ImageNet/result/experiment_results.json](ImageNet/result/experiment_results.json)。

结果通常会包含：

- `task`
- `best_val_acc`
- `test_acc`



## 项目二：YOLO + VisDrone 检测与跟踪

这个项目包含 VisDrone 数据集的下载、转换、训练、跟踪推理和结果可视化，覆盖完整流程。

### 主要文件

- [scripts/download_visdrone_kaggle.py](YOLO/scripts/download_visdrone_kaggle.py)：下载并解压 VisDrone 数据
- [scripts/convert_visdrone_det_to_yolo.py](YOLO/scripts/convert_visdrone_det_to_yolo.py)：把 VisDrone 标注转成 YOLO 格式
- [scripts/run_train.py](YOLO/scripts/run_train.py)：训练检测模型
- [scripts/run_test_tracking.py](YOLO/scripts/run_test_tracking.py)：对视频或视频目录做跟踪并输出结果
- [scripts/track_compact_labels.py](YOLO/scripts/track_compact_labels.py)：生成更紧凑的跟踪标签视频
- [scripts/make_visdrone_clip.py](YOLO/scripts/make_visdrone_clip.py)：从图片序列合成视频
- [scripts/images_to_video.py](YOLO/scripts/images_to_video.py)：把图片目录按顺序转成视频

公共工具位于：

- [scripts/common/visdrone.py](YOLO/scripts/common/visdrone.py)
- [scripts/common/video_utils.py](YOLO/scripts/common/video_utils.py)
- [scripts/common/io_utils.py](YOLO/scripts/common/io_utils.py)

### 依赖安装

```bash
cd YOLO
pip install -r requirements.txt
```

### 完整流程

#### 1. 配置 Kaggle API

先在 Kaggle 账户里生成 `kaggle.json`，放到 Windows 的 `%USERPROFILE%\.kaggle\kaggle.json`。

#### 2. 下载 VisDrone 数据

```bash
python scripts/download_visdrone_kaggle.py --out_dir data/raw
```

#### 3. 转成 YOLO 格式

```bash
python scripts/convert_visdrone_det_to_yolo.py --raw_root data/raw --out_root data/visdrone_yolo
```

转换后通常会得到：

- `data/visdrone_yolo/images/train`
- `data/visdrone_yolo/labels/train`
- `data/visdrone_yolo/images/val`
- `data/visdrone_yolo/labels/val`
- `visdrone.yaml`

#### 4. 训练检测模型

```bash
python scripts/run_train.py --model yolov8n.pt --data visdrone.yaml --epochs 100 --imgsz 896 --batch 4 --device 0
```

脚本默认开启更适合小目标任务的配置，包括：

- `--multi_scale`
- `--cos_lr`
- `--close_mosaic 15`
- `--patience 50`

训练输出会保存到 [YOLO/train_model/](YOLO/train_model/) 下的子目录中，同时最新权重会复制为 [YOLO/train_model/best.pt](YOLO/train_model/best.pt)。

#### 5. 准备测试视频

如果没有现成视频，可以从 VisDrone 图片序列合成：

```bash
python scripts/make_visdrone_clip.py --images_dir data/raw/VisDrone2019-DET-train/images --out output/visdrone_20s.mp4 --seconds 20 --fps 25
```

#### 6. 运行跟踪

```bash
python scripts/run_test_tracking.py --model train_model/train-1 --source input/video.mp4 --count_line 965,0,965,2000
```

`--model` 可以直接传 `train_model`、`train_model/train-1`，也可以传某个 `best.pt` 文件。

每次运行会自动创建新的输出目录，例如 [YOLO/output/test1/](YOLO/output/test1/) 和 [YOLO/output/test2/](YOLO/output/test2/)。

### 跟踪标签

如果密集场景里标签太挤，可以用更紧凑的版本：

```bash
python scripts/track_compact_labels.py --model train_model/train-1/weights/best.pt --source input/test.mp4 --tracker trackers/botsort_reid_antiswitch.yaml --out output/compact_track.mp4 --conf 0.25 --output_mode id_cls --min_label_area 1200 --font_scale 0.34
```

`output_mode` 可选：

- `id_cls`：ID + 类别
- `id`：仅 ID
- `cls`：仅类别

### 常见问题

如果出现 ID 互换，可以尝试：

1. 改用 `trackers/botsort_reid_antiswitch.yaml`
2. 适当提高 `--conf`
3. 用同一个视频对比不同 tracker 的稳定性

如果标签遮挡严重，优先调大 `--min_label_area`，或者直接使用 [scripts/track_compact_labels.py](YOLO/scripts/track_compact_labels.py)。

## 输出目录汇总

- [ImageNet/result/](ImageNet/result/)：分类实验结果 JSON
- [Unet/results/](Unet/results/)：分割训练权重、曲线、汇总和 test 评估结果
- [YOLO/train_model/](YOLO/train_model/)：检测训练权重
- [YOLO/output/](YOLO/output/)：跟踪视频与可视化结果


## 项目三：Unet 语义分割实验

这是 Oxford-IIIT Pet 的语义分割项目，目标是把图像分成 3 类 trimap，并比较不同损失函数的效果。

### 主要文件

- [train.py](Unet/train.py)：训练入口
- [eval_test.py](Unet/eval_test.py)：在官方 test 集上评估训练好的模型
- [data.py](Unet/data.py)：数据读取与 DataLoader 构建
- [model.py](Unet/model.py)：U-Net 网络结构
- [losses.py](Unet/losses.py)：`ce`、`dice`、`combined` 三种损失
- [metrics.py](Unet/metrics.py)：mIoU 指标

### 依赖安装

```bash
pip install torch numpy pillow matplotlib wandb
```

第二条同样是可选项。

### 数据准备

默认会优先使用仓库父目录下的 `data/oxford-iiit-pet`。如果本地没有，会尝试自动下载 Oxford-IIIT Pet 数据集。

数据目录应包含：

- `images/`
- `annotations/trimaps/`
- `annotations/trainval.txt`
- `annotations/test.txt`

### 训练方式

```bash
cd Unet
python train.py
```

单独跑某个损失函数：

```bash
python train.py --loss-mode ce
python train.py --loss-mode dice
python train.py --loss-mode combined
```

一次跑完三组实验：

```bash
python train.py --loss-mode all
```

常用参数：

- `--epochs`：训练轮数
- `--batch-size`：批大小
- `--image-size`：输入尺寸
- `--num-workers`：DataLoader 线程数
- `--lr`：学习率
- `--base-channels`：U-Net 基础通道数
- `--loss-mode`：`ce` / `dice` / `combined` / `all`
- `--output-dir`：输出目录
- `--use-wandb`：启用 W&B 记录

### 评估方式

训练完成后运行：

```bash
python eval_test.py
```

该脚本会读取 [Unet/results/](Unet/results/) 下各损失对应的 `best.pt`，在官方 test 集上计算 mIoU，并生成测试汇总。

### 输出结果

默认会在 [Unet/results/](Unet/results/) 下保存：

- `best.pt`
- `history.json`
- `curves.png`
- `predictions.png`
- `summary.csv`
- `summary.json`
- `miou_comparison.png`
- `test_summary.csv`
- `test_summary.json`
- `test_miou_comparison.png`