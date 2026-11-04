"""
迁移学习策略模块。

支持的策略：
    1. Linear Probing：冻结全部特征提取层，只训练分类头。
    2. Partial Fine-tuning：冻结前面的网络层，微调最后若干层 + 分类头。
    3. Full Fine-tuning：微调整个模型。

实现思路：
    先冻结所有参数，再按策略逐层解冻，保证「哪些参数可训练」完全可控、
    可复现，并且能准确统计可训练参数量。
"""

import torch.nn as nn


def _set_train_mode_with_frozen_bn(model):
    """将模型置为 train 模式，但已冻结的 BatchNorm 保持 eval。

    防止 Linear Probing 时 model.train() 把冻结特征提取器里的 BN
    running_mean / running_var 意外更新。
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # 该 BN 层所有参数均冻结 → 保持 eval
            if all(not p.requires_grad for p in m.parameters()):
                m.eval()


def _set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad


def _unfreeze_head(model):
    """解冻最后的分类 Linear 层（ResNet 的 fc / EfficientNet 的 classifier）。"""
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        _set_requires_grad(last_linear, True)


def _unfreeze_last_blocks(model, n):
    """解冻特征提取器最后 n 个「块」。"""
    if n <= 0:
        return
    if hasattr(model, "layer4") and hasattr(model, "layer3"):
        # ResNet：layer4 -> layer3 -> layer2 -> layer1
        names = ["layer4", "layer3", "layer2", "layer1"][:n]
        for name in names:
            _set_requires_grad(getattr(model, name), True)
    elif hasattr(model, "features") and hasattr(model, "classifier"):
        # EfficientNet：features 是 Sequential，解冻最后 n 个 block
        children = list(model.features.children())
        for block in children[-n:]:
            _set_requires_grad(block, True)
    else:
        # 兜底：解冻最后 n 个顶层子模块
        children = list(model.children())
        for block in children[-n:]:
            _set_requires_grad(block, True)


def apply_strategy(model, strategy, unfreeze_blocks=2):
    """
    把微调策略应用到模型上（原地修改并返回同一模型）。

    参数:
        model: 已加载（并替换分类头）的模型。
        strategy: "linear_probing" | "partial_finetune" | "full_finetune"。
        unfreeze_blocks: partial_finetune 时解冻的特征提取块数量。

    返回:
        model
    """
    # 1) 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    strategy = strategy.lower()
    if strategy == "linear_probing":
        _unfreeze_head(model)
    elif strategy == "full_finetune":
        for p in model.parameters():
            p.requires_grad = True
    elif strategy == "partial_finetune":
        _unfreeze_head(model)
        _unfreeze_last_blocks(model, int(unfreeze_blocks))
    else:
        raise ValueError(
            f"不支持的微调策略: {strategy}，"
            "可选: linear_probing / partial_finetune / full_finetune"
        )
    return model


def summarize_trainable(model):
    """返回「可训练/冻结」参数个数的简明摘要，便于日志与对接报告。"""
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    return {"trainable_params": trainable, "frozen_params": frozen}
