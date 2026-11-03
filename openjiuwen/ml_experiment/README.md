# ml_experiment —— ResearchPilot 机器学习实验模块

机器学习实验与数据模块，为科研 Agent 提供可调用的 PyTorch 实验能力。

> 研究方案：
> **An Empirical Study of Transfer Learning Strategies for Few-Shot Image Classification with Pretrained Vision Models**
> 基于预训练视觉模型（ResNet50 / EfficientNet-B0）的小样本图像分类迁移学习策略对比，数据集 Oxford Flowers 102。

---

## 一、目录结构

```
ml_experiment/
├── datasets/
│   ├── __init__.py
│   └── dataset_loader.py       # 数据加载 / 预处理 / 增强 / 分层抽样 / DataLoader
├── models/
│   ├── __init__.py
│   └── model_loader.py         # 预训练模型加载 / 替换分类头 / 参数统计
├── strategies/
│   ├── __init__.py
│   └── finetune.py             # Linear Probing / Partial / Full Fine-tuning
├── training/
│   ├── __init__.py
│   └── trainer.py              # 训练循环 / 验证 / 调度 / 早停 / 最优模型
├── evaluation/
│   ├── __init__.py
│   └── metrics.py              # Accuracy / Precision / Recall / F1
├── results/
│   ├── __init__.py
│   └── result_saver.py         # 结果保存 / 曲线 / checkpoint / 汇总 CSV
├── configs/
│   ├── default.json            # 基线：Full Fine-tuning / ResNet50 / 100%
│   ├── smoke_test.json         # 冒烟测试（少量数据、2 epoch，先跑这个验证环境）
│   ├── full_finetune.json
│   ├── linear_probing.json
│   ├── partial_finetune.json
│   ├── data_ratio_0.10.json
│   ├── data_ratio_0.25.json
│   ├── data_ratio_0.50.json
│   └── efficientnet_b0.json
├── logs/                       # 训练日志（运行时生成）
├── checkpoints/                # 模型权重（运行时生成）
├── main.py                     # 命令行入口
├── interface.py                # openJiuwen Agent 调用接口（run_experiment）
└── requirements.txt
```

运行时还会生成：
- `results/runs/<experiment_id>/`：配置快照、结果 JSON、指标 JSON、history CSV、曲线图。
- `results/experiments_summary.csv`：跨实验汇总表（供论文与对比用）。
- `../data/`（相对 ml_experiment）：Oxford Flowers 102 数据集下载目录。

---

## 二、环境配置

硬件：NVIDIA RTX 4060 Laptop（8GB 显存），Python 3.13。

```bash
python --version                                   # 应为 3.13
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

安装依赖：

```bash
pip install -r requirements.txt
```

> 环境已在本机验证通过（Python 3.13.2 / torch 2.14.1+cu126 / CUDA 可用）。
> 代码已在入口处设置 `KMP_DUPLICATE_LIB_OK=TRUE`，避免 Windows 上 torch 与
> matplotlib 的 OpenMP 重复加载冲突（否则训练完成后绘图阶段会崩溃）。

---

## 三、快速开始

所有命令都在 `ml_experiment` 目录下执行（保证相对路径正确）。

**第 0 步：准备数据集（只需一次）**

```bash
cd <项目根目录>          # 即 openjiuwen 目录
python prepare_dataset.py
```

官方源 www.robots.ox.ac.uk 在国内无法访问，此脚本改用 fast.ai 镜像（内容与官方一致），
下载并解压到 `data/oxford-102-flowers/`（约 330MB）。

**第 1 步：冒烟测试（验证环境与代码跑通）**

```bash
cd ml_experiment
python main.py --config configs/smoke_test.json
```

冒烟测试用 10% 数据、2 个 epoch，约 1 分钟跑完（本机已实测通过）。

**第 2 步：正式基线实验**

```bash
python main.py --config configs/default.json
```

**第 3 步：完整实验矩阵**（见第六节）

```bash
python main.py --config configs/linear_probing.json
python main.py --config configs/partial_finetune.json
python main.py --config configs/full_finetune.json
python main.py --config configs/data_ratio_0.10.json
python main.py --config configs/data_ratio_0.25.json
python main.py --config configs/data_ratio_0.50.json
python main.py --config configs/efficientnet_b0.json   # 时间允许时
```

> Windows 提示：如果训练时出现 multiprocessing 相关报错，把对应 JSON 里的 `num_workers` 改为 `0`（smoke_test.json 已经是 0）。

---

## 四、JSON 配置说明

配置采用「扁平结构」，未写出的字段会使用 `interface.DEFAULT_CONFIG` 里的默认值。核心字段：

| 字段 | 含义 | 可选值 / 默认 |
| :--- | :--- | :--- |
| `experiment_id` | 实验编号（输出目录名，必填） | 字符串 |
| `experiment_name` | 实验可读名称 | 字符串 |
| `dataset` | 数据集 | `oxford_flowers102` |
| `model` | 预训练模型 | `resnet50` / `efficientnet_b0` |
| `pretrained` | 是否加载 ImageNet 预训练 | `true` / `false` |
| `strategy` | 迁移学习策略 | `linear_probing` / `partial_finetune` / `full_finetune` |
| `partial_unfreeze_blocks` | Partial 微调解冻的块数 | 整数（默认 2） |
| `data_ratio` | 训练数据比例 | `0.1` / `0.25` / `0.5` / `1.0` |
| `epochs` | 训练轮数 | 整数 |
| `batch_size` | 批大小 | 整数 |
| `learning_rate` | 学习率 | 浮点 |
| `weight_decay` | 权重衰减 | 浮点 |
| `optimizer` | 优化器 | `adamw` / `adam` / `sgd` |
| `scheduler` | 学习率调度 | `cosine` / `step` / `none` |
| `early_stopping_patience` | 早停耐心（0 为关闭） | 整数 |
| `seed` | 随机种子 | 整数 |

修改 JSON 即可改变模型 / 策略 / 数据比例，无需改动训练代码主体。

---

## 五、openJiuwen Agent 调用接口

对外入口在 `interface.py`，Agent 只需调用：

```python
import sys
sys.path.insert(0, "/path/to/openjiuwen/ml_experiment")
from interface import run_experiment

result = run_experiment("configs/full_finetune.json")
print(result["status"])        # "success" | "error"
print(result["test_metrics"])  # 测试集指标
print(result["outputs"]["metrics_file"])  # 指标文件路径
```

- 输入：JSON 配置文件路径（或 `config_override` 覆盖部分字段）。
- 输出：结构化 dict，含指标、资源统计、输出文件路径。详细结构见 `interface.py` 的 docstring。
- 也可以 `from interface import run_experiment_from_config`，直接传配置 dict。

---

## 六、实验矩阵

| 实验 | 配置文件 | 模型 | 策略 | 数据比例 |
| :--- | :--- | :--- | :--- | :--- |
| 基线 | `default.json` | ResNet50 | Full Fine-tuning | 100% |
| 冒烟测试 | `smoke_test.json` | ResNet50 | Full Fine-tuning | 10% |
| 实验一-策略对比 | `linear_probing.json` | ResNet50 | Linear Probing | 100% |
| 实验一-策略对比 | `partial_finetune.json` | ResNet50 | Partial Fine-tuning | 100% |
| 实验一-策略对比 | `full_finetune.json` | ResNet50 | Full Fine-tuning | 100% |
| 实验二-数据比例 | `data_ratio_0.10.json` | ResNet50 | Full Fine-tuning | 10% |
| 实验二-数据比例 | `data_ratio_0.25.json` | ResNet50 | Full Fine-tuning | 25% |
| 实验二-数据比例 | `data_ratio_0.50.json` | ResNet50 | Full Fine-tuning | 50% |
| 实验三-模型对比（可选） | `efficientnet_b0.json` | EfficientNet-B0 | Full Fine-tuning | 100% |

实验记录要求：每次实验都会记录实验编号、模型、数据集、数据划分、策略、种子、epoch、batch size、学习率、Accuracy/Precision/Recall/F1、训练时间、GPU 显存、代码版本（Git commit）。测试集绝不用于调参；不同实验使用一致的数据划分（固定 seed）。

---

## 七、复现指南

1. 确认 PyTorch/CUDA 可用。
2. 安装 `requirements.txt`。
3. `cd ml_experiment`。
4. 按第六节运行对应配置文件（`python main.py --config configs/xxx.json`）。
5. 结果在 `results/runs/<experiment_id>/`，汇总表在 `results/experiments_summary.csv`。
6. 复现验证：在干净环境重复第 4 步，比对 `results.json` 与 `experiments_summary.csv` 中的指标。

---

## 八、「动态代码生成」衔接（后续阶段）

预设工具（本模块）优先保证稳定性；当任务超出预设能力时，由 Agent 的 Code Generator 生成 PyTorch 代码、Code Reviewer 检查、在受限环境中执行。本模块暴露的 `interface.run_experiment` 可作为「固定工具」被 Agent 选择调用，动态代码执行另由 Agent 实现（本阶段只预留接口）。
