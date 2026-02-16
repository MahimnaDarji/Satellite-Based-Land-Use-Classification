from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

def _check_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {s}")

def _min_class_count(df: pd.DataFrame) -> int:
    return int(df["label"].value_counts().min())


def make_splits(
    index_csv: Path,
    out_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[Path, Path, Path, Path]:
    _check_ratios(train_ratio, val_ratio, test_ratio)

    index_csv = index_csv.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not index_csv.exists():
        raise FileNotFoundError(f"Index file not found: {index_csv}")

    df = pd.read_csv(index_csv)
    required_cols = {"path", "raw_class", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Index file missing columns: {sorted(missing)}")

    df = df.dropna(subset=["path", "label"]).copy()
    df["label"] = df["label"].astype(str)

    min_cnt = _min_class_count(df)
    if min_cnt < 3:
        raise RuntimeError(
            f"Too few samples in at least one final class (min={min_cnt}). Splitting will be unstable."
        )

    test_size = test_ratio
    remaining = 1.0 - test_size
    val_size_relative = val_ratio / remaining

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_relative,
        random_state=seed,
        stratify=train_val_df["label"],
    )

    train_df = train_df.sort_values(["label", "raw_class", "path"]).reset_index(drop=True)
    val_df = val_df.sort_values(["label", "raw_class", "path"]).reset_index(drop=True)
    test_df = test_df.sort_values(["label", "raw_class", "path"]).reset_index(drop=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    all_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    ).sort_values(["split", "label", "raw_class", "path"]).reset_index(drop=True)

    all_path = out_dir / "splits_all.csv"
    all_df.to_csv(all_path, index=False)

    print("\nSPLIT COMPLETE")
    print(f"Index:  {index_csv}")
    print(f"Out:    {out_dir}")
    print(f"Train:  {len(train_df)}")
    print(f"Val:    {len(val_df)}")
    print(f"Test:   {len(test_df)}")

    def _counts(x: pd.DataFrame) -> pd.Series:
        return x["label"].value_counts().sort_index()

    print("\nPer-class counts (train/val/test):")
    merged = pd.concat(
        [
            _counts(train_df).rename("train"),
            _counts(val_df).rename("val"),
            _counts(test_df).rename("test"),
        ],
        axis=1,
    ).fillna(0).astype(int)

    for label, row in merged.iterrows():
        print(f"  {label}: {row['train']}/{row['val']}/{row['test']}")

    print("")
    return train_path, val_path, test_path, all_path

if __name__ == "__main__":
    make_splits(
        index_csv=Path("datasets/processed/index.csv"),
        out_dir=Path("datasets/splits"),
        seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
