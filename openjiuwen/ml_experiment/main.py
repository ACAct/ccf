"""
实验入口。

用法（在 ml_experiment 目录下执行）：

    python main.py --config configs/default.json

参数:
    --config       JSON 配置文件路径（默认 configs/default.json）

说明:
    main.py 只是命令行入口，真正的逻辑都在 interface.run_experiment 中，
    方便被 openJiuwen Agent 复用同一个入口。
"""

import argparse
import json
import sys

from interface import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="ResearchPilot 机器学习实验入口")
    parser.add_argument(
        "--config", type=str, default="configs/default.json",
        help="JSON 配置文件路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_experiment(args.config)

    if result["status"] == "success":
        print("\n[完成] 实验:", result["experiment_id"])
        print("[验证集最优 Accuracy]:", result["best_val_acc"])
        print("[测试集指标]:")
        print(json.dumps(result["test_metrics"], ensure_ascii=False, indent=2))
        print("[输出文件]:")
        for k, v in result["outputs"].items():
            print(f"  {k}: {v}")
    else:
        print("[失败] 实验执行出错:", result["error"])
        print(result["traceback"])
        sys.exit(1)


if __name__ == "__main__":
    main()
