"""
数据集加载模块。

职责：
    1. 加载数据集（当前支持 Oxford Flowers 102）。
    2. 数据预处理与数据增强。
    3. 划分训练集 / 验证集 / 测试集。
    4. 支持「小样本」数据比例（data_ratio）下的按类别分层抽样。
    5. 创建 DataLoader。

数据集来源说明：
    官方源 www.robots.ox.ac.uk 在国内网络无法访问（连接超时），因此本模块
    读取 fast.ai 提供的镜像（内容与官方一致，含官方 train/val/test 划分与标签）。
    镜像由仓库根目录的 prepare_dataset.py 下载并解压为：

        data/oxford-102-flowers/
            jpg/image_00001.jpg ... image_08189.jpg   (8189 张图)
            train.txt   (1020 行："jpg/image_XXXXX.jpg <0 起标签>")
            valid.txt   (1020 行)
            test.txt    (6149 行)

约定：
    - 沿用官方 train / val / test 划分，保证不同实验之间数据划分完全一致（可复现）。
    - data_ratio 只对训练集做「按类别分层抽样」，val / test 保持完整，
      测试集绝不参与调参。
    - 使用固定随机种子，保证不同实验之间抽样结果一致。
"""

import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

# ImageNet 预训练模型使用的归一化统计量
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 支持的数据集：名称 -> 元信息（类别数、简介、链接）
DATASET_META = {
    "oxford_flowers102": {
        "num_classes": 102,
        "description": "Oxford 102 Category Flower Dataset",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
    },
}


class OxfordFlowers102(Dataset):
    """
    Oxford Flowers 102 数据集加载器（读取 fast.ai 镜像的目录结构）。

    目录结构：
        <root>/oxford-102-flowers/jpg/image_XXXXX.jpg
        <root>/oxford-102-flowers/{train,valid,test}.txt
    其中每个 txt 文件每行格式为 "jpg/image_XXXXX.jpg <0 起标签>"。
    """

    _split_files = {"train": "train.txt", "val": "valid.txt", "test": "test.txt"}

    def __init__(self, root, split, transform=None):
        if split not in self._split_files:
            raise ValueError(f"split 只能是 train/val/test，收到: {split}")

        self.root = Path(root) / "oxford-102-flowers"
        split_file = self.root / self._split_files[split]
        if not split_file.exists():
            raise FileNotFoundError(
                f"未找到数据集划分文件 {split_file}。"
                "请先在仓库根目录运行: python prepare_dataset.py"
            )

        self.samples = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(" ", 1)
            self.samples.append((self.root / rel_path, int(label)))

        self._labels = [label for _, label in self.samples]  # 供分层抽样使用
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.loader(str(path))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_num_classes(dataset_name):
    """返回数据集的类别数。"""
    meta = DATASET_META.get(dataset_name)
    if meta is None:
        raise ValueError(
            f"不支持的数据集: {dataset_name}，可选: {list(DATASET_META)}"
        )
    return meta["num_classes"]


def get_transforms(image_size, is_train):
    """
    返回数据预处理 / 增强变换。

    训练集：随机裁剪 + 水平翻转 + 颜色抖动（轻量数据增强）。
    验证/测试集：Resize + CenterCrop（固定、无随机性，保证评估稳定）。
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_dataset(dataset_name, split, data_dir, transform):
    """根据名称与 split 构建原始数据集。"""
    if dataset_name == "oxford_flowers102":
        return OxfordFlowers102(root=data_dir, split=split, transform=transform)
    raise ValueError(f"不支持的数据集: {dataset_name}")


def _get_labels(dataset):
    """尽量从数据集内部取出标签列表，用于分层抽样。"""
    if hasattr(dataset, "_labels") and dataset._labels is not None:
        return list(dataset._labels)
    if hasattr(dataset, "targets") and dataset.targets is not None:
        return list(dataset.targets)
    # 兜底：逐样本读取标签（较慢，仅在不暴露标签属性的数据集上使用）
    return [int(dataset[i][1]) for i in range(len(dataset))]


def stratified_subsample(dataset, ratio, seed):
    """
    对数据集做「按类别分层抽样」，返回子集。

    目的：实现 few-shot / 小样本比例实验，保证每个类别都按相同比例保留，
    并且使用固定 seed 保证不同实验之间抽样结果一致（可复现）。

    返回:
        Subset 或原 dataset（ratio >= 1.0 时原样返回）。
    """
    if ratio >= 1.0:
        return dataset
    if ratio <= 0.0:
        raise ValueError("data_ratio 必须大于 0")

    labels = np.asarray(_get_labels(dataset))
    rng = np.random.default_rng(seed)
    indices = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        k = max(1, int(round(len(cls_idx) * ratio)))
        chosen = rng.choice(cls_idx, size=k, replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices).astype(int)
    indices.sort()
    return Subset(dataset, indices.tolist())


def build_dataloaders(config):
    """
    根据配置构建 train / val / test 三个 DataLoader。

    参数:
        config: 合并默认值后的完整配置 dict。

    返回:
        (train_loader, val_loader, test_loader, dataset_sizes)
        dataset_sizes = {"train": n, "val": n, "test": n}
    """
    dataset_name = config.get("dataset", "oxford_flowers102")
    data_dir = config.get("data_dir", "../data")
    image_size = int(config.get("image_size", 224))
    data_ratio = float(config.get("data_ratio", 1.0))
    seed = int(config.get("seed", 42))
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 4))
    pin_memory = bool(config.get("pin_memory", True))

    train_ds = build_dataset(
        dataset_name, "train", data_dir, get_transforms(image_size, True)
    )
    val_ds = build_dataset(
        dataset_name, "val", data_dir, get_transforms(image_size, False)
    )
    test_ds = build_dataset(
        dataset_name, "test", data_dir, get_transforms(image_size, False)
    )

    # 小样本比例：只对训练集做分层抽样
    train_ds = stratified_subsample(train_ds, data_ratio, seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    sizes = {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}
    return train_loader, val_loader, test_loader, sizes
