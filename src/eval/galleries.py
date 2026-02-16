from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

@dataclass(frozen=True)
class GalleryConfig:
    image_size: int = 224
    cols: int = 5
    max_images: int = 25
    dpi: int = 200

def _load_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((size, size))
        return np.asarray(im)

def _draw_grid(images: List[np.ndarray], titles: List[str], out_path: Path, cols: int, dpi: int) -> None:
    n = len(images)
    if n == 0:
        raise ValueError("No images to draw")

    rows = ceil(n / cols)
    fig = plt.figure(figsize=(cols * 3, rows * 3), dpi=dpi)

    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(images[i])
        ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def _build_test_view(splits_dir: Path, preds_csv: Path) -> pd.DataFrame:
    test_df = pd.read_csv(Path(splits_dir) / "test.csv")
    preds = pd.read_csv(preds_csv)

    if len(test_df) != len(preds):
        raise RuntimeError(
            f"Row mismatch: test.csv has {len(test_df)} rows but predictions has {len(preds)} rows. "
            "This means ordering/alignment broke."
        )

    df = test_df.copy()
    df["true_label"] = preds["true_label"].astype(str)
    df["pred_label"] = preds["pred_label"].astype(str)
    df["confidence"] = preds["confidence"].astype(float)
    df["is_correct"] = df["true_label"] == df["pred_label"]
    return df

def make_gallery(df: pd.DataFrame, out_path: Path, cfg: GalleryConfig, seed: int) -> None:
    df = df.sample(frac=1.0, random_state=seed).head(cfg.max_images).reset_index(drop=True)

    images: List[np.ndarray] = []
    titles: List[str] = []

    for _, r in df.iterrows():
        img = _load_rgb(Path(r["path"]), cfg.image_size)
        images.append(img)

        pct = int(round(float(r["confidence"]) * 100))
        titles.append(f"True: {r['true_label']} | Pred: {r['pred_label']} ({pct}%)")

    _draw_grid(images, titles, out_path, cols=cfg.cols, dpi=cfg.dpi)

def main() -> None:
    preds_csv = Path("outputs/phase3/test_predictions.csv")
    splits_dir = Path("datasets/splits")
    out_dir = Path("outputs/phase3")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = GalleryConfig(image_size=224, cols=3, max_images=9, dpi=220)
    df = _build_test_view(splits_dir=splits_dir, preds_csv=preds_csv)

    make_gallery(
        df=df,
        out_path=out_dir / "prediction_gallery_all.png",
        cfg=cfg,
        seed=42,
    )

    correct_df = df[df["is_correct"]].copy()
    if correct_df.empty:
        raise RuntimeError("No correct predictions found. Something is seriously wrong.")

    make_gallery(
        df=correct_df,
        out_path=out_dir / "prediction_gallery_correct.png",
        cfg=cfg,
        seed=42,
    )

    wrong_df = df[~df["is_correct"]].copy()
    if wrong_df.empty:
        raise RuntimeError("No misclassifications found. Nothing to plot.")

    make_gallery(
        df=wrong_df,
        out_path=out_dir / "misclassification_gallery.png",
        cfg=cfg,
        seed=42,
    )

    print("\nGALLERIES COMPLETE")
    print(f"Saved: {out_dir / 'prediction_gallery_all.png'}")
    print(f"Saved: {out_dir / 'prediction_gallery_correct.png'}")
    print(f"Saved: {out_dir / 'misclassification_gallery.png'}\n")

if __name__ == "__main__":
    main()
