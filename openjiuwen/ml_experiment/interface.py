"""
机器学习实验调用接口（供 openJiuwen Agent 调用）。

这是本模块对外暴露的唯一入口。Agent 无需关心内部实现，
只需调用 run_experiment()。

典型用法（Agent 侧）：

    import sys
    sys.path.insert(0, "/path/to/openjiuwen/ml_experiment")  # 项目根下的 ml_experiment
    from interface import run_experiment

    result = run_experiment("configs/full_finetune.json")
    if result["status"] == "success":
        print(result["test_metrics"])
        print(result["outputs"]["metrics_file"])

run_experiment 的返回结构见函数 docstring。
"""

import os

# 解决 Windows 上 torch(自带 libiomp5md.dll) 与 matplotlib/numpy 的 OpenMP
# 重复加载冲突，避免训练完成后绘图阶段崩溃。必须在 import torch/numpy/matplotlib 之前设置。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import logging
import platform
import random
import subprocess
import sys
import traceback

import numpy as np
import torch

from datasets.dataset_loader import build_dataloaders, get_num_classes
from models.model_loader import (
    get_model,
    move_to_device,
    count_total_parameters,
    count_trainable_parameters,
    SUPPORTED_MODELS,
)
from strategies.finetune import apply_strategy
from training.trainer import Trainer
from evaluation.metrics import compute_metrics
from results.result_saver import save_experiment_outputs, append_summary_csv


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "dataset": "oxford_flowers102",
    "data_dir": "../data",
    "image_size": 224,
    "num_classes": 102,
    "data_ratio": 1.0,
    "num_workers": 4,
    "pin_memory": True,
    "model": "resnet50",
    "pretrained": True,
    "strategy": "full_finetune",
    "partial_unfreeze_blocks": 2,
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "label_smoothing": 0.0,
    "early_stopping_patience": 0,
    "seed": 42,
    "results_dir": "results/runs",
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
    "summary_csv": "results/experiments_summary.csv",
}


def set_seed(seed):
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 尽量保证卷积算子确定性（可能略微降低速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """自动选择 GPU，没有 GPU 时回退到 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(config_path):
    """读取 JSON 配置并与默认值合并。"""
    with open(config_path, "r", encoding="utf-8") as f:
        user_config = json.load(f)
    config = dict(DEFAULT_CONFIG)
    config.update(user_config)
    return config


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------

SUPPORTED_STRATEGIES = {"linear_probing", "partial_finetune", "full_finetune"}
SUPPORTED_DATASETS = {"oxford_flowers102"}
SUPPORTED_OPTIMIZERS = {"adamw", "adam", "sgd"}
SUPPORTED_SCHEDULERS = {"cosine", "step", "none", "off", ""}


def validate_config(config):
    """校验合并后的实验配置；失败时抛出含具体字段的 ValueError。"""
    # ── 必需字段 ──
    experiment_id = config.get("experiment_id", "")
    if not experiment_id:
        raise ValueError("experiment_id 不能为空")

    # ── 数值范围 ──
    epochs = config.get("epochs", 20)
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f"epochs 必须为正整数，得到: {epochs}")

    batch_size = config.get("batch_size", 32)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，得到: {batch_size}")

    lr = config.get("learning_rate", 1e-4)
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError(f"learning_rate 必须 > 0，得到: {lr}")

    data_ratio = config.get("data_ratio", 1.0)
    if not isinstance(data_ratio, (int, float)) or not (0 < data_ratio <= 1):
        raise ValueError(f"data_ratio 必须在 (0, 1] 内，得到: {data_ratio}")

    seed = config.get("seed", 42)
    try:
        int(seed)
    except (ValueError, TypeError):
        raise ValueError(f"seed 必须为合法整数，得到: {seed}")

    # ── 枚举字段 ──
    model = config.get("model", "")
    if model.lower() not in SUPPORTED_MODELS:
        raise ValueError(
            f"不支持的 model: {model}，可选: {sorted(SUPPORTED_MODELS)}"
        )

    strategy = config.get("strategy", "")
    if strategy.lower() not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"不支持的 strategy: {strategy}，可选: {sorted(SUPPORTED_STRATEGIES)}"
        )

    dataset = config.get("dataset", "")
    if dataset.lower() not in SUPPORTED_DATASETS:
        raise ValueError(
            f"不支持的 dataset: {dataset}，可选: {sorted(SUPPORTED_DATASETS)}"
        )

    optimizer = config.get("optimizer", "adamw")
    if optimizer.lower() not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"不支持的 optimizer: {optimizer}，可选: {sorted(SUPPORTED_OPTIMIZERS)}"
        )

    scheduler = config.get("scheduler", "cosine")
    if scheduler.lower() not in SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"不支持的 scheduler: {scheduler}，可选: {sorted(SUPPORTED_SCHEDULERS)}"
        )


def _setup_logger(config):
    """配置控制台 + 文件日志，返回 logger。"""
    experiment_id = config.get("experiment_id", "experiment")
    log_dir = config.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_id}.log")

    logger = logging.getLogger(f"ml_experiment.{experiment_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# 环境信息采集
# ---------------------------------------------------------------------------

def _get_git_commit():
    """返回 HEAD commit 短哈希；非 git 仓库或获取失败时返回 None。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def collect_environment_info():
    """采集当前运行环境信息，返回 dict；获取失败不抛异常。"""
    info = {
        "python_version": sys.version.split(" ", 1)[0] if sys.version else None,
        "torch_version": torch.__version__,
        "torchvision_version": None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": None,
        "os": platform.platform(),
        "git_commit": _get_git_commit(),
    }
    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    try:
        import torchvision
        info["torchvision_version"] = torchvision.__version__
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# 核心接口
# ---------------------------------------------------------------------------

def run_experiment(config_path, config_override=None):
    """
    读取实验配置并执行机器学习实验（openJiuwen Agent 对接入口）。

    参数:
        config_path: JSON 配置文件路径（相对当前工作目录或绝对路径）。
        config_override: 可选 dict，覆盖配置文件中的部分字段。
                         Agent 若在内存中动态生成配置，可传 {} 并 override。

    返回:
        dict:
            status: "success" | "error"
            experiment_id: 实验编号
            config: 实际使用的完整配置（合并默认值后）
            dataset_sizes: {"train": n, "val": n, "test": n}
            best_val_acc: 验证集最优 Accuracy
            best_epoch: 最优 epoch
            val_metrics: 最优 epoch 验证集指标
            test_metrics: 测试集指标
            resource: {"total_params", "trainable_params",
                       "total_train_time_sec", "avg_epoch_time_sec",
                       "peak_gpu_memory_mb"}
            outputs: 各输出文件路径（metrics_file / results_file /
                     history_file / checkpoint / plots 等）
    """
    config = None
    try:
        config = load_config(config_path)
        if config_override:
            config.update(config_override)

        # ── 配置合法性校验 ──
        validate_config(config)

        experiment_id = config.get("experiment_id", "experiment")
        seed = int(config.get("seed", 42))
        set_seed(seed)

        logger = _setup_logger(config)
        device = get_device()

        logger.info("=" * 70)
        logger.info("开始实验: %s", experiment_id)
        logger.info("设备: %s", device)
        if device.type == "cuda":
            logger.info("GPU: %s", torch.cuda.get_device_name(device))
            logger.info("CUDA 可用: %s", torch.cuda.is_available())

        # 1) 数据
        num_classes = int(config.get("num_classes", get_num_classes(config.get("dataset"))))
        config["num_classes"] = num_classes
        train_loader, val_loader, test_loader, dataset_sizes = build_dataloaders(config)
        logger.info("数据集划分: %s", dataset_sizes)

        # 2) 模型 + 微调策略
        model = get_model(config.get("model"), num_classes, config.get("pretrained", True))
        apply_strategy(model, config.get("strategy"), config.get("partial_unfreeze_blocks", 2))
        model = move_to_device(model, device)

        total_params = count_total_parameters(model)
        trainable_params = count_trainable_parameters(model)
        resource_info = {"total_params": total_params, "trainable_params": trainable_params}
        logger.info(
            "模型: %s | 总参数 %d | 可训练参数 %d | 策略 %s",
            config.get("model"), total_params, trainable_params, config.get("strategy"),
        )

        # 3) 训练
        trainer = Trainer(model, train_loader, val_loader, device, config, num_classes, logger)
        run_summary = trainer.run()

        # 4) 测试集评估（用验证集最优权重）
        if run_summary.get("best_state") is not None:
            model.load_state_dict(run_summary["best_state"])
        test_loss, test_metrics = trainer.evaluate(test_loader)

        val_metrics = {"accuracy": run_summary["best_val_acc"], "epoch": run_summary["best_epoch"]}
        be = run_summary["best_epoch"]
        if be >= 1:
            val_metrics = {
                "epoch": be,
                "accuracy": run_summary["history"]["val_acc"][be - 1],
                "f1_macro": run_summary["history"]["val_f1_macro"][be - 1],
                "loss": run_summary["history"]["val_loss"][be - 1],
            }
        logger.info("测试集指标: %s", test_metrics)

        # 5) 保存输出
        env_info = collect_environment_info()
        outputs = save_experiment_outputs(
            config, run_summary, val_metrics, test_metrics, dataset_sizes,
            resource_info, env_info,
        )

        # 6) 汇总 CSV（跨实验对比）
        summary_row = {
            "experiment_id": experiment_id,
            "experiment_name": config.get("experiment_name", ""),
            "model": config.get("model"),
            "strategy": config.get("strategy"),
            "data_ratio": config.get("data_ratio"),
            "epochs": len(run_summary["history"]["train_loss"]),
            "batch_size": config.get("batch_size"),
            "learning_rate": config.get("learning_rate"),
            "seed": seed,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "best_val_acc": run_summary["best_val_acc"],
            "test_accuracy": test_metrics.get("accuracy"),
            "test_f1_macro": test_metrics.get("f1_macro"),
            "test_precision_macro": test_metrics.get("precision_macro"),
            "test_recall_macro": test_metrics.get("recall_macro"),
            "total_train_time_sec": round(run_summary["total_train_time_sec"], 3),
            "train_time_sec": round(run_summary.get("train_time_sec", 0), 3),
            "val_time_sec": round(run_summary.get("val_time_sec", 0), 3),
            "peak_gpu_memory_mb": round(run_summary["peak_gpu_memory_mb"], 3),
        }
        append_summary_csv(summary_row, config.get("summary_csv", "results/experiments_summary.csv"))

        logger.info(
            "实验完成: %s | best_val_acc=%.4f | test_acc=%.4f",
            experiment_id, run_summary["best_val_acc"], test_metrics["accuracy"],
        )
        logger.info("输出目录: %s", outputs["output_dir"])

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "config": config,
            "dataset_sizes": dataset_sizes,
            "best_val_acc": run_summary["best_val_acc"],
            "best_epoch": run_summary["best_epoch"],
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "resource": {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "total_train_time_sec": run_summary["total_train_time_sec"],
                "train_time_sec": run_summary.get("train_time_sec", 0),
                "val_time_sec": run_summary.get("val_time_sec", 0),
                "avg_epoch_time_sec": run_summary["avg_epoch_time_sec"],
                "peak_gpu_memory_mb": run_summary["peak_gpu_memory_mb"],
            },
            "outputs": outputs,
        }
    except Exception as exc:
        tb = traceback.format_exc()
        try:
            logging.getLogger("ml_experiment").error("实验失败:\n%s", tb)
        except Exception:
            pass
        return {
            "status": "error",
            "experiment_id": config.get("experiment_id", "unknown") if config else "unknown",
            "error": str(exc),
            "traceback": tb,
        }


def run_experiment_from_config(config_dict, tmp_config_path="configs/_generated_by_agent.json"):
    """
    Agent 直接传入配置 dict 时使用：写入临时文件后调用 run_experiment。

    参数:
        config_dict: 完整的实验配置 dict（至少需要 experiment_id）。
        tmp_config_path: 临时配置文件的存放路径。
    """
    os.makedirs(os.path.dirname(tmp_config_path) or ".", exist_ok=True)
    with open(tmp_config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    return run_experiment(tmp_config_path)
