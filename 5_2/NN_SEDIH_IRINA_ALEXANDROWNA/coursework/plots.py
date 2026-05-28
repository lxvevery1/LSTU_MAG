"""Parse results.txt and generate training plots (loss and accuracy curves)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_results(path: str | Path) -> dict[str, dict[str, list[float]]]:
    """Parse training output from results.txt.

    Returns:
        {"standard": {"loss": [...], "test_acc": [...]},
         "separable": {"loss": [...], "test_acc": [...]}}
    """
    text = Path(path).read_text(encoding="utf-8")

    data: dict[str, dict[str, list[float]]] = {}
    current: str | None = None

    loss_pat = re.compile(r"loss=([\d.]+)")
    acc_pat = re.compile(r"test_acc=([\d.]+)%")

    for line in text.splitlines():
        if "Training standard" in line:
            current = "standard"
            data[current] = {"loss": [], "test_acc": []}
        elif "Training depthwise-separable" in line:
            current = "separable"
            data[current] = {"loss": [], "test_acc": []}

        if current is None:
            continue

        m_loss = loss_pat.search(line)
        m_acc = acc_pat.search(line)

        if m_loss and m_acc:
            data[current]["loss"].append(float(m_loss.group(1)))
            data[current]["test_acc"].append(float(m_acc.group(1)))

    return data


def plot_training(data: dict[str, dict[str, list[float]]], output_dir: str | Path):
    """Generate loss and accuracy plots, save as PDF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs_std = range(1, len(data["standard"]["loss"]) + 1)
    epochs_sep = range(1, len(data["separable"]["loss"]) + 1)

    # --- loss plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_std, data["standard"]["loss"], "b-o", markersize=4,
            label="Standard ResNet-18")
    ax.plot(epochs_sep, data["separable"]["loss"], "r-s", markersize=4,
            label="Depthwise Separable ResNet-18")

    # Mark LR drop point
    milestones = [10, 15]
    for m in milestones:
        ax.axvline(x=m, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Training Loss")
    ax.set_title("Функция потерь (Cross-Entropy) по эпохам")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_plot.pdf", dpi=150)
    plt.close(fig)

    # --- accuracy plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_std, data["standard"]["test_acc"], "b-o", markersize=4,
            label="Standard ResNet-18")
    ax.plot(epochs_sep, data["separable"]["test_acc"], "r-s", markersize=4,
            label="Depthwise Separable ResNet-18")

    for m in milestones:
        ax.axvline(x=m, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Точность на тестовой выборке по эпохам")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "acc_plot.pdf", dpi=150)
    plt.close(fig)

    print(f"Saved: {(output_dir / 'loss_plot.pdf')}")
    print(f"Saved: {(output_dir / 'acc_plot.pdf')}")


def main():
    results_path = Path("results.txt")
    if not results_path.exists():
        print(f"Error: {results_path} not found", file=sys.stderr)
        sys.exit(1)

    data = parse_results(results_path)
    plot_training(data, Path("pdf"))


if __name__ == "__main__":
    main()
