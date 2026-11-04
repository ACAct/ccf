"""
训练模块。

职责：
    1. 单 epoch 训练（前向 + 损失 + 反向 + 参数更新）。
    2. 验证集评估。
    3. 学习率调度。
    4. 早停（可选）。
    5. 最优模型（按验证集 Accuracy）保存。
    6. 统计训练时间、每个 epoch 时间、GPU 显存峰值。

指标依赖 evaluation.metrics.compute_metrics。
"""

import copy
import logging
import time

import torch
import torch.nn as nn

from evaluation.metrics import compute_metrics
from strategies.finetune import _set_train_mode_with_frozen_bn


def _resolve_optimizer(name):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW
    if name == "adam":
        return torch.optim.Adam
    if name == "sgd":
        return torch.optim.SGD
    raise ValueError(f"不支持的优化器: {name}，可选: adamw / adam / sgd")


class Trainer:
    def __init__(
        self, model, train_loader, val_loader, device, config, num_classes,
        logger=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.num_classes = num_classes
        self.logger = logger or logging.getLogger(__name__)

        label_smoothing = float(config.get("label_smoothing", 0.0))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.best_val_acc = 0.0
        self.best_epoch = -1
        self.best_state = None
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
            "lr": [],
            "epoch_time_sec": [],
            "train_time_sec": [],
            "val_time_sec": [],
        }

    def _build_optimizer(self):
        opt_cls = _resolve_optimizer(self.config.get("optimizer", "adamw"))
        lr = float(self.config.get("learning_rate", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-4))
        params = [p for p in self.model.parameters() if p.requires_grad]
        kwargs = {"lr": lr, "weight_decay": wd}
        if opt_cls is torch.optim.SGD:
            kwargs["momentum"] = 0.9
        return opt_cls(params, **kwargs)

    def _build_scheduler(self):
        sched = self.config.get("scheduler", "cosine").lower()
        epochs = int(self.config.get("epochs", 20))
        if sched in ("none", "off", ""):
            return None
        if sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        if sched == "step":
            step_size = int(self.config.get("scheduler_step_size", 10))
            gamma = float(self.config.get("scheduler_gamma", 0.1))
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        raise ValueError(
            f"不支持的 scheduler: {sched}，可选: cosine / step / none"
        )

    def train_epoch(self):
        _set_train_mode_with_frozen_bn(self.model)
        running_loss, running_correct, total = 0.0, 0, 0
        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            total += images.size(0)

        return running_loss / total, running_correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        running_loss, total = 0.0, 0
        all_preds, all_targets = [], []
        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_targets.append(targets.cpu())
            total += images.size(0)

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        metrics = compute_metrics(targets, preds, self.num_classes)
        return running_loss / total, metrics

    def run(self):
        """
        执行完整训练流程，返回结果摘要 dict。

        返回:
            dict: history / best_val_acc / best_epoch / best_state / final_state
                  / total_train_time_sec / avg_epoch_time_sec / peak_gpu_memory_mb
        """
        config = self.config
        epochs = int(config.get("epochs", 20))
        patience = int(config.get("early_stopping_patience", 0))

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        total_start = time.time()
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            ep_start = time.time()

            # ── 训练 ──
            train_start = time.time()
            train_loss, train_acc = self.train_epoch()
            train_time = time.time() - train_start

            # ── 验证 ──
            val_start = time.time()
            val_loss, val_metrics = self.evaluate(self.val_loader)
            val_time = time.time() - val_start

            ep_time = time.time() - ep_start

            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1_macro"].append(val_metrics["f1_macro"])
            self.history["lr"].append(lr)
            self.history["epoch_time_sec"].append(ep_time)
            self.history["train_time_sec"].append(train_time)
            self.history["val_time_sec"].append(val_time)

            # 更新最优模型（按验证集 Accuracy）
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            self.logger.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | "
                "val_loss=%.4f val_acc=%.4f val_f1=%.4f | lr=%.2e | %.1fs",
                epoch, epochs, train_loss, train_acc,
                val_loss, val_metrics["accuracy"], val_metrics["f1_macro"],
                lr, ep_time,
            )

            if self.scheduler is not None:
                self.scheduler.step()

            if patience > 0 and epochs_no_improve >= patience:
                self.logger.info(
                    "早停触发：连续 %d 个 epoch 验证集未提升，停止训练。",
                    epochs_no_improve,
                )
                break

        total_time = time.time() - total_start
        n_epochs = len(self.history["train_loss"])
        avg_epoch_time = total_time / n_epochs if n_epochs else 0.0
        total_train_time = sum(self.history.get("train_time_sec", []))
        total_val_time = sum(self.history.get("val_time_sec", []))

        peak_memory_mb = 0.0
        if self.device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        return {
            "history": self.history,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "best_state": self.best_state,
            "final_state": copy.deepcopy(self.model.state_dict()),
            "total_train_time_sec": total_time,
            "train_time_sec": total_train_time,
            "val_time_sec": total_val_time,
            "avg_epoch_time_sec": avg_epoch_time,
            "peak_gpu_memory_mb": peak_memory_mb,
        }
