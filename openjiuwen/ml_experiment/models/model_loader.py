"""
模型加载模块。

职责：
    1. 加载预训练模型（ResNet50 / EfficientNet-B0）。
    2. 替换分类层以适配目标类别数。
    3. 将模型移动到 GPU / CPU。
    4. 统计可训练参数与总参数。

说明：
    - 使用 torchvision 的 weights 接口加载 ImageNet 预训练权重。
"""

import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS = {"resnet50", "efficientnet_b0"}


def get_model(model_name, num_classes, pretrained=True):
    """根据名称加载（预训练）模型，并替换分类头为 num_classes 输出。"""
    model_name = model_name.lower()
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b0":
        weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"不支持的模型: {model_name}，可选: {sorted(SUPPORTED_MODELS)}"
        )
    return model


def move_to_device(model, device):
    """把模型移动到指定设备并返回。"""
    return model.to(device)


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
