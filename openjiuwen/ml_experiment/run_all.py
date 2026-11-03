"""
顺序运行全部正式实验。

用法（在 ml_experiment 目录下执行）：
    python run_all.py

按顺序执行：
    1) full_finetune        实验一 · 微调策略对比（100% 数据）
    2) linear_probing       实验一 · 微调策略对比
    3) partial_finetune     实验一 · 微调策略对比
    4) data_ratio_0.10      实验二 · 数据比例
    5) data_ratio_0.25      实验二 · 数据比例
    6) data_ratio_0.50      实验二 · 数据比例
    7) efficientnet_b0      实验三 · 模型对比

每个实验都会写入 results/experiments_summary.csv，结果与权重存到
results/runs/<实验id>/ 和 checkpoints/<实验id>/。
"""

import subprocess
import sys
import time

CONFIGS = [
    "full_finetune",
    "linear_probing",
    "partial_finetune",
    "data_ratio_0.10",
    "data_ratio_0.25",
    "data_ratio_0.50",
    "efficientnet_b0",
]


def main():
    total = len(CONFIGS)
    results = []
    for i, name in enumerate(CONFIGS, 1):
        bar = "=" * 70
        print(f"\n{bar}\n[RUNNER] {i}/{total} START {name}\n{bar}", flush=True)
        start = time.time()
        ret = subprocess.run(
            [sys.executable, "main.py", "--config", f"configs/{name}.json"]
        )
        elapsed = time.time() - start
        status = "OK" if ret.returncode == 0 else "FAIL"
        results.append((name, status, elapsed))
        print(
            f"\n[RUNNER] {i}/{total} DONE {name} -> {status} "
            f"(exit={ret.returncode}, elapsed={elapsed/60:.1f}min)",
            flush=True,
        )

    print("\n" + "=" * 70, flush=True)
    print("[RUNNER] 全部结束，汇总：", flush=True)
    for name, status, elapsed in results:
        print(f"  {name:20s} {status:4s} {elapsed/60:6.1f} min", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
