"""Training ResNet-18 on CIFAR-10."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import ResNet, resnet18


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def train_model(
    model: ResNet,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int = 50,
    lr: float = 0.1,
    device: torch.device | None = None,
    save_path: Path | None = None,
) -> dict[str, float]:
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1
    )

    best_acc = 0.0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        print(
            f"Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
            f"test_acc={test_acc:.2f}%  best={best_acc:.2f}%"
        )

    elapsed = time.perf_counter() - t0
    final_acc = evaluate(model, test_loader, device)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "separable": model.separable,
                "test_acc": final_acc,
                "best_acc": best_acc,
            },
            save_path,
        )

    return {
        "final_acc": final_acc,
        "best_acc": best_acc,
        "train_time_s": elapsed,
    }


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> ResNet:
    device = device or torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = resnet18(separable=bool(ckpt.get("separable", False)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
