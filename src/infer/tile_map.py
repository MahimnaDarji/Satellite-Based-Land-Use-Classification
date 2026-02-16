from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from src.training.datamodule import build_label_maps
from src.training.model import ModelConfig, build_model


@dataclass(frozen=True)
class TileConfig:
    tile_size: int = 224
    stride: int = 224
    batch_size: int = 64


def _get_palette(n: int) -> List[Tuple[int, int, int]]:
    base = [
        (34, 139, 34),
        (220, 20, 60),
        (30, 144, 255),
        (255, 215, 0),
        (160, 82, 45),
        (148, 0, 211),
        (0, 206, 209),
        (255, 140, 0),
        (105, 105, 105),
        (255, 105, 180),
    ]
    if n <= len(base):
        return base[:n]
    out = base[:]
    rng = np.random.default_rng(123)
    while len(out) < n:
        out.append(tuple(int(x) for x in rng.integers(0, 256, size=3)))
    return out[:n]


def _load_model(ckpt: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    cfg = ModelConfig(model_name="resnet18", num_classes=num_classes, pretrained=False, dropout=0.2)
    model = build_model(cfg)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _ensure_min_size(img: Image.Image, tile_size: int) -> Image.Image:
    w, h = img.size
    if w >= tile_size and h >= tile_size:
        return img
    new_w = max(w, tile_size)
    new_h = max(h, tile_size)
    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def _extract_tiles(img: Image.Image, tile_size: int, stride: int) -> Tuple[List[Tuple[int, int]], List[Image.Image]]:
    img = _ensure_min_size(img, tile_size)
    w, h = img.size

    coords: List[Tuple[int, int]] = []
    tiles: List[Image.Image] = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            coords.append((x, y))
            tiles.append(img.crop((x, y, x + tile_size, y + tile_size)))

    if not tiles:
        coords = [(0, 0)]
        tiles = [img.crop((0, 0, tile_size, tile_size))]

    return coords, tiles


@torch.no_grad()
def _predict_tiles(
    model: torch.nn.Module,
    tiles: List[Image.Image],
    device: torch.device,
    tile_size: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    tf = transforms.Compose(
        [
            transforms.Resize((tile_size, tile_size)),
            transforms.ToTensor(),
        ]
    )

    preds: List[int] = []
    confs: List[float] = []

    for i in range(0, len(tiles), batch_size):
        batch = tiles[i : i + batch_size]
        x = torch.stack([tf(t.convert("RGB")) for t in batch], dim=0).to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        p = torch.argmax(probs, dim=1)
        c = probs.max(dim=1).values

        preds.extend(p.cpu().numpy().tolist())
        confs.extend(c.cpu().numpy().tolist())

    return np.array(preds, dtype=np.int32), np.array(confs, dtype=np.float32)


def _build_map_image(
    base_img: Image.Image,
    coords: List[Tuple[int, int]],
    pred_idx: np.ndarray,
    palette: List[Tuple[int, int, int]],
    tile_size: int,
    alpha: int,
) -> Image.Image:
    base_img = _ensure_min_size(base_img, tile_size)
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for (x, y), cls in zip(coords, pred_idx.tolist()):
        color = palette[int(cls)]
        draw.rectangle([x, y, x + tile_size, y + tile_size], fill=(color[0], color[1], color[2], alpha))

    return Image.alpha_composite(base_img.convert("RGBA"), overlay)


def _add_legend(
    img: Image.Image,
    labels: List[str],
    palette: List[Tuple[int, int, int]],
    box: int = 22,
    pad: int = 10,
) -> Image.Image:
    w, h = img.size
    legend_w = 360
    legend_h = pad * 2 + len(labels) * (box + 8)

    canvas = Image.new("RGBA", (w + legend_w, max(h, legend_h)), (255, 255, 255, 255))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    x0 = w + pad
    y0 = pad

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    draw.text((x0, y0), "Legend", fill=(0, 0, 0, 255), font=font)
    y = y0 + 26

    for i, lbl in enumerate(labels):
        color = palette[i]
        draw.rectangle([x0, y, x0 + box, y + box], fill=(color[0], color[1], color[2], 255))
        draw.text((x0 + box + 10, y + 2), lbl, fill=(0, 0, 0, 255), font=font)
        y += box + 8

    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", type=str, required=True)
    ap.add_argument("--tile-size", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=110)
    ap.add_argument("--out-dir", type=str, default="outputs/phase5")
    args = ap.parse_args()

    input_path = Path(args.input_image)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_csv = Path("datasets/processed/index.csv")
    labels, _, _ = build_label_maps(index_csv)

    ckpt = Path("models/checkpoints/best_model.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = _load_model(ckpt, num_classes=len(labels), device=device)

    base = Image.open(input_path).convert("RGB")
    coords, tiles = _extract_tiles(base, tile_size=args.tile_size, stride=args.stride)

    pred_idx, conf = _predict_tiles(
        model=model,
        tiles=tiles,
        device=device,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
    )

    palette = _get_palette(len(labels))
    mapped = _build_map_image(
        base_img=base,
        coords=coords,
        pred_idx=pred_idx,
        palette=palette,
        tile_size=args.tile_size,
        alpha=int(args.alpha),
    )

    final = _add_legend(mapped, labels=labels, palette=palette)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_map = out_dir / "landuse_overlay_map.png"
    final.save(out_map)

    np.save(out_dir / "tile_pred_idx.npy", pred_idx)
    np.save(out_dir / "tile_conf.npy", conf)

    meta = {
        "input_image": str(input_path.as_posix()),
        "tile_size": int(args.tile_size),
        "stride": int(args.stride),
        "num_tiles": int(len(pred_idx)),
        "labels": labels,
    }
    (out_dir / "meta.txt").write_text(str(meta), encoding="utf-8")

    print("\nTILE MAP COMPLETE")
    print(f"Saved: {out_map}")
    print(f"Tiles: {len(pred_idx)}")
    print(f"Out:   {out_dir}\n")


if __name__ == "__main__":
    main()
