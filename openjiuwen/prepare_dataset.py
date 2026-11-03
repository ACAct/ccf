"""
下载并准备 Oxford Flowers 102 数据集。

说明：
    官方源 www.robots.ox.ac.uk 在当前网络（国内）无法访问（连接超时）。
    本脚本改用 fast.ai 提供的镜像 tgz（内容与官方一致，含官方 train/val/test 划分与标签），
    解压成 ml_experiment/datasets 可读取的目录结构：

        data/oxford-102-flowers/
            jpg/image_00001.jpg ... image_08189.jpg   (8189 张图)
            train.txt   (1020 行："jpg/image_XXXXX.jpg <0 起标签>")
            valid.txt   (1020 行)
            test.txt    (6149 行)

用法：
    python prepare_dataset.py
"""

import os
import sys
import tarfile
import time
import urllib.request

URL = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-102-flowers.tgz"
TGZ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "oxford-102-flowers.tgz")
DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download():
    os.makedirs(os.path.dirname(TGZ_PATH), exist_ok=True)
    if os.path.exists(TGZ_PATH) and os.path.getsize(TGZ_PATH) > 0:
        print("已存在 tgz，跳过下载:", TGZ_PATH)
        return
    print("下载数据集镜像:", URL)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        start = time.time()
        with open(TGZ_PATH, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100.0 / total
                    speed = done / max(time.time() - start, 1e-6) / 1024 / 1024
                    sys.stdout.write(f"\r{done/1024/1024:.1f}/{total/1024/1024:.1f} MB ({pct:.1f}%) {speed:.1f} MB/s")
                    sys.stdout.flush()
    print("\n下载完成:", TGZ_PATH)


def extract():
    print("解压中 ...")
    with tarfile.open(TGZ_PATH, "r:gz") as tf:
        tf.extractall(DEST_DIR, filter="data")
    print("解压完成 ->", os.path.join(DEST_DIR, "oxford-102-flowers"))


if __name__ == "__main__":
    download()
    extract()
    print("数据集准备完毕。")
