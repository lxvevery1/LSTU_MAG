"""CPU latency and model size comparison."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from models import count_parameters, model_size_bytes


@dataclass
class BenchmarkResult:
    name: str
    params: int
    size_mb: float
    latency_ms: float
    latency_std_ms: float
    test_acc: float | None = None


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    *,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    warmup: int = 20,
    repeats: int = 100,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """Mean and std of single-batch inference time in milliseconds."""
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()

    x = torch.randn(*input_shape, device=device)
    for _ in range(warmup):
        model(x)

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000.0)

    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    return mean, variance**0.5


def benchmark_model(
    model: nn.Module,
    name: str,
    *,
    test_acc: float | None = None,
    device: torch.device | None = None,
    warmup: int = 20,
    repeats: int = 100,
) -> BenchmarkResult:
    device = device or torch.device("cpu")
    model_cpu = model.to(device)
    model_cpu.eval()

    lat_mean, lat_std = measure_latency(
        model_cpu,
        warmup=warmup,
        repeats=repeats,
        device=device,
    )
    return BenchmarkResult(
        name=name,
        params=count_parameters(model_cpu),
        size_mb=model_size_bytes(model_cpu) / (1024 * 1024),
        latency_ms=lat_mean,
        latency_std_ms=lat_std,
        test_acc=test_acc,
    )


def format_table(results: list[BenchmarkResult]) -> str:
    headers = (
        "Model",
        "Params",
        "Size (MB)",
        "Latency (ms)",
        "Test acc (%)",
    )
    rows: list[tuple[str, ...]] = []
    for r in results:
        acc = f"{r.test_acc:.2f}" if r.test_acc is not None else "—"
        rows.append(
            (
                r.name,
                f"{r.params:,}",
                f"{r.size_mb:.3f}",
                f"{r.latency_ms:.3f} ± {r.latency_std_ms:.3f}",
                acc,
            )
        )

    col_widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt_row(cells: tuple[str, ...]) -> str:
        return "  ".join(cells[i].ljust(col_widths[i]) for i in range(len(cells)))

    sep = "  ".join("-" * w for w in col_widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)
