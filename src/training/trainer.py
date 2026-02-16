from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.training.datamodule import DataConfig, build_label_maps, build_loaders
from src.training.model import ModelConfig, build_model

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = Path("models/checkpoints")
    history_path: Path = Path("models/history.csv")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer=None,
    device="cpu",
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy_from_logits(logits, y) * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_acc / total_samples

def train() -> None:
    data_cfg = DataConfig()
    index_csv = Path("datasets/processed/index.csv")
    labels, label_to_idx, _ = build_label_maps(index_csv)

    train_loader, val_loader, _ = build_loaders(data_cfg, label_to_idx)

    model_cfg = ModelConfig(
        model_name="resnet18",
        num_classes=len(labels),
        pretrained=True,
        dropout=0.2,
    )

    model = build_model(model_cfg)

    train_cfg = TrainConfig()
    device = torch.device(train_cfg.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=train_cfg.lr)

    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_records = []

    best_val_acc = 0.0
    best_path = train_cfg.checkpoint_dir / "best_model.pt"

    print("\nTRAINING STARTED")
    print(f"Device: {device}")
    print(f"Epochs: {train_cfg.epochs}\n")

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer=None, device=device
        )

        history_records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    history_df = pd.DataFrame(history_records)
    train_cfg.history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(train_cfg.history_path, index=False)

    print("\nTRAINING COMPLETE")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Best model saved at: {best_path}")
    print(f"History saved at: {train_cfg.history_path}\n")

if __name__ == "__main__":
    train()
