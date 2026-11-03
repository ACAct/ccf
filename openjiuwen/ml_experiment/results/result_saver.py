"""
结果保存模块。

职责：
    1. 保存实验配置（复现用）。
    2. 保存实验结果 JSON（指标、时间、显存、参数统计）。
    3. 保存每 epoch 的历史 CSV。
    4. 绘制并保存 Loss 曲线 / Accuracy 曲线。
    5. 保存模型权重（checkpoint）。
    6. 追加一行到汇总 CSV（跨实验对比）。

输出目录约定：
    results/runs/<experiment_id>/
        config.json          实验配置快照
        results.json         指标 + 资源统计
        metrics.json         验证/测试指标（供 Agent / 论文模块直接读取）
        history.csv          逐 epoch 历史
        loss_curve.png       Loss 曲线
        accuracy_curve.png   Accuracy 曲线
    checkpoints/<experiment_id>/
        best_model.pth       验证集最优权重
        last_model.pth       最后一次权重
"""

import csv
import json
import os

import torch


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data, path):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def save_checkpoint(state_dict, path):
    _ensure_dir(os.path.dirname(path))
    torch.save(state_dict, path)
    return path


def save_history_csv(history, path):
    """把逐 epoch 历史写成 CSV，列为 epoch + 各指标。"""
    keys = list(history.keys())
    n = len(history[keys[0]]) if keys else 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(n):
            row = [i + 1] + [history[k][i] for k in keys]
            writer.writerow(row)
    return path


def save_plots(history, output_dir):
    """绘制 Loss 与 Accuracy 曲线。matplotlib 缺失时优雅降级（不报错）。"""
    paths = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return paths

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["loss_curve"] = loss_path

    # Accuracy 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["accuracy_curve"] = acc_path

    return paths


def append_summary_csv(row, csv_path):
    """向汇总 CSV 追加一行实验记录（用于实验对比与论文数据）。"""
    _ensure_dir(os.path.dirname(csv_path))
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return csv_path


def save_experiment_outputs(
    config, run_summary, val_metrics, test_metrics, dataset_sizes, resource_info=None
):
    """
    保存一次实验的全部输出，返回各文件的路径字典。

    参数:
        config: 合并后的完整配置 dict。
        run_summary: Trainer.run() 返回的摘要 dict。
        val_metrics: 最优 epoch 对应的验证集指标 dict。
        test_metrics: 测试集指标 dict。
        dataset_sizes: {"train": n, "val": n, "test": n}。
        resource_info: {"total_params": n, "trainable_params": n}。
    """
    resource_info = resource_info or {}
    experiment_id = config.get("experiment_id", "experiment")
    results_dir = config.get("results_dir", "results/runs")
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")

    out_dir = _ensure_dir(os.path.join(results_dir, experiment_id))
    ckpt_dir = _ensure_dir(os.path.join(checkpoint_dir, experiment_id))

    history = run_summary["history"]

    # 1) 配置快照（保证可复现）
    config_path = save_json(config, os.path.join(out_dir, "config.json"))

    # 2) 结果 JSON（指标 + 资源统计 + 数据规模）
    results = {
        "experiment_id": experiment_id,
        "experiment_name": config.get("experiment_name", ""),
        "dataset_sizes": dataset_sizes,
        "best_val_acc": run_summary["best_val_acc"],
        "best_epoch": run_summary["best_epoch"],
        "total_train_time_sec": run_summary["total_train_time_sec"],
        "avg_epoch_time_sec": run_summary["avg_epoch_time_sec"],
        "peak_gpu_memory_mb": run_summary["peak_gpu_memory_mb"],
        "trainable_params": resource_info.get("trainable_params"),
        "total_params": resource_info.get("total_params"),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "num_epochs_trained": len(history["train_loss"]),
    }
    results_path = save_json(results, os.path.join(out_dir, "results.json"))

    # 3) 指标 JSON（供 Agent / 论文模块直接读取）
    metrics_path = save_json(
        {"val": val_metrics, "test": test_metrics},
        os.path.join(out_dir, "metrics.json"),
    )

    # 4) 历史 CSV
    history_path = save_history_csv(history, os.path.join(out_dir, "history.csv"))

    # 5) 曲线图
    plot_paths = save_plots(history, out_dir)

    # 6) checkpoint
    checkpoint_paths = {}
    if run_summary.get("best_state") is not None:
        checkpoint_paths["best_model"] = save_checkpoint(
            run_summary["best_state"], os.path.join(ckpt_dir, "best_model.pth")
        )
    if run_summary.get("final_state") is not None:
        checkpoint_paths["last_model"] = save_checkpoint(
            run_summary["final_state"], os.path.join(ckpt_dir, "last_model.pth")
        )

    return {
        "output_dir": out_dir,
        "checkpoint_dir": ckpt_dir,
        "config_file": config_path,
        "results_file": results_path,
        "metrics_file": metrics_path,
        "history_file": history_path,
        "plots": plot_paths,
        "checkpoints": checkpoint_paths,
    }
