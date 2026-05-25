#!/usr/bin/env python3
"""Train ResNet-18 (standard vs depthwise-separable) on CIFAR-10 and compare."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from benchmark import benchmark_model, format_table
from models import resnet18
from train import get_dataloaders, train_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ResNet-18 vs depthwise-separable ResNet-18 on CIFAR-10"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--benchmark-warmup", type=int, default=20)
    p.add_argument("--benchmark-repeats", type=int, default=100)
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Only benchmark existing checkpoints",
    )
    p.add_argument(
        "--device-train",
        default="auto",
        help="Training device: auto, cpu, or cuda",
    )
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cpu = torch.device("cpu")
    train_device = cpu if args.skip_train else resolve_device(args.device_train)
    torch.set_num_threads(args.cpu_threads)

    ckpt_std = args.output_dir / "resnet18_standard.pt"
    ckpt_ds = args.output_dir / "resnet18_separable.pt"

    if not args.skip_train:
        train_loader, test_loader = get_dataloaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        print("=== Training standard ResNet-18 ===")
        std_stats = train_model(
            resnet18(separable=False),
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=train_device,
            save_path=ckpt_std,
        )

        print("\n=== Training depthwise-separable ResNet-18 ===")
        ds_stats = train_model(
            resnet18(separable=True),
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=train_device,
            save_path=ckpt_ds,
        )
    else:
        std_stats = ds_stats = {}

    print("\n=== CPU benchmark (batch=1, 32×32) ===")

    models_info = [
        ("ResNet-18 (standard)", ckpt_std, False),
        ("ResNet-18 (depthwise-separable)", ckpt_ds, True),
    ]

    bench_results = []
    report: dict = {"training": {}, "benchmark": []}

    for name, path, separable in models_info:
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            acc = float(ckpt.get("test_acc", 0))
            model = resnet18(separable=separable)
            model.load_state_dict(ckpt["state_dict"])
        else:
            print(f"Warning: checkpoint missing {path}, using untrained weights")
            acc = None
            model = resnet18(separable=separable)

        br = benchmark_model(
            model,
            name,
            test_acc=acc,
            device=cpu,
            warmup=args.benchmark_warmup,
            repeats=args.benchmark_repeats,
        )
        bench_results.append(br)
        report["benchmark"].append(
            {
                "name": br.name,
                "params": br.params,
                "size_mb": br.size_mb,
                "latency_ms": br.latency_ms,
                "latency_std_ms": br.latency_std_ms,
                "test_acc": br.test_acc,
            }
        )

    print(format_table(bench_results))

    if std_stats:
        report["training"]["standard"] = std_stats
    if ds_stats:
        report["training"]["separable"] = ds_stats

    if bench_results[0].test_acc is not None and bench_results[1].test_acc is not None:
        speedup = bench_results[0].latency_ms / bench_results[1].latency_ms
        size_ratio = bench_results[0].size_mb / bench_results[1].size_mb
        acc_drop = bench_results[0].test_acc - bench_results[1].test_acc
        print(
            f"\nSeparable vs standard: "
            f"latency {speedup:.2f}× faster, "
            f"size {size_ratio:.2f}× smaller, "
            f"accuracy Δ {acc_drop:+.2f} pp"
        )
        report["summary"] = {
            "latency_speedup": speedup,
            "size_ratio": size_ratio,
            "accuracy_delta_pp": acc_drop,
            "cpu_threads": args.cpu_threads,
            "benchmark_warmup": args.benchmark_warmup,
            "benchmark_repeats": args.benchmark_repeats,
            "seed": args.seed,
        }

    out_json = args.output_dir / "results.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
