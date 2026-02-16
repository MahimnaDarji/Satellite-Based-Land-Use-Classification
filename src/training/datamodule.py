from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class LandCoverCSVDataset(Dataset):
    def __init__(self, csv_path: Path, label_to_idx: Dict[str, int], image_size: int, augment: bool):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain columns: path, label")

        self.df["path"] = self.df["path"].astype(str)
        self.df["label"] = self.df["label"].astype(str)

        self.label_to_idx = dict(label_to_idx)
        unknown = sorted(set(self.df["label"].unique().tolist()) - set(self.label_to_idx.keys()))
        if unknown:
            raise ValueError(f"CSV contains labels not in label_to_idx: {unknown}")

        base = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        if augment:
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
            ]
            self.tf = transforms.Compose(aug + base)
        else:
            self.tf = transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = Path(row["path"])
        label = row["label"]
        y = self.label_to_idx[label]

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            x = self.tf(im)

        return x, y

@dataclass(frozen=True)
class DataConfig:
    splits_dir: Path = Path("datasets/splits")
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42

def build_label_maps(index_csv: Path) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    df = pd.read_csv(index_csv)
    if "label" not in df.columns:
        raise ValueError("index.csv must contain column: label")
    labels = sorted(df["label"].astype(str).unique().tolist())
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    return labels, label_to_idx, idx_to_label

def build_loaders(cfg: DataConfig, label_to_idx: Dict[str, int]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_csv = cfg.splits_dir / "train.csv"
    val_csv = cfg.splits_dir / "val.csv"
    test_csv = cfg.splits_dir / "test.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_ds = LandCoverCSVDataset(train_csv, label_to_idx, image_size=cfg.image_size, augment=True)
    val_ds = LandCoverCSVDataset(val_csv, label_to_idx, image_size=cfg.image_size, augment=False)
    test_ds = LandCoverCSVDataset(test_csv, label_to_idx, image_size=cfg.image_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

def sanity_check_one_batch(train_loader: DataLoader, labels: List[str]) -> None:
    x, y = next(iter(train_loader))
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise RuntimeError("Unexpected batch types")

    print("\nDATALOADER SANITY CHECK")
    print(f"Batch x shape: {tuple(x.shape)}")
    print(f"Batch y shape: {tuple(y.shape)}")
    print(f"Min label idx: {int(y.min())}, Max label idx: {int(y.max())}")
    print(f"Num classes: {len(labels)}")
    print(f"Example labels: {labels[:min(10, len(labels))]}")
    print("")

if __name__ == "__main__":
    index_csv = Path("datasets/processed/index.csv")
    labels, label_to_idx, _ = build_label_maps(index_csv)

    cfg = DataConfig(
        splits_dir=Path("datasets/splits"),
        image_size=224,
        batch_size=16,
        num_workers=2,
        seed=42,
    )

    train_loader, _, _ = build_loaders(cfg, label_to_idx)
    sanity_check_one_batch(train_loader, labels)
