"""
指标评估模块。

计算指标：
    - Accuracy
    - Precision (macro / weighted)
    - Recall (macro / weighted)
    - F1-score (macro / weighted)

实现说明：
    直接基于 NumPy 混淆矩阵计算，不依赖 sklearn，
    减少环境依赖，便于在干净环境中复现。
"""

import numpy as np


def _confusion_matrix(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    idx = y_true * num_classes + y_pred
    cm = np.bincount(
        idx, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    return cm


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)


def compute_metrics(y_true, y_pred, num_classes):
    """
    计算多分类指标。

    参数:
        y_true: 真实标签（array-like）。
        y_pred: 预测标签（array-like）。
        num_classes: 类别数。

    返回:
        dict: accuracy / precision_macro / recall_macro / f1_macro
              / precision_weighted / recall_weighted / f1_weighted
    """
    y_true = np.asarray(y_true).ravel().astype(np.int64)
    y_pred = np.asarray(y_pred).ravel().astype(np.int64)

    cm = _confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.float64)

    total = cm.sum()
    accuracy = float(tp.sum() / total) if total > 0 else 0.0

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    macro_p = float(precision.mean())
    macro_r = float(recall.mean())
    macro_f1 = float(f1.mean())

    support_total = support.sum()
    weighted_p = float((precision * support).sum() / support_total) if support_total > 0 else 0.0
    weighted_r = float((recall * support).sum() / support_total) if support_total > 0 else 0.0
    weighted_f1 = float((f1 * support).sum() / support_total) if support_total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1,
        "precision_weighted": weighted_p,
        "recall_weighted": weighted_r,
        "f1_weighted": weighted_f1,
    }
