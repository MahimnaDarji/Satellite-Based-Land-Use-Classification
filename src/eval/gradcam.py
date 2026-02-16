from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.training.datamodule import build_label_maps
from src.training.model import ModelConfig, build_model

@dataclass(frozen=True)
class GradCAMConfig:
    image_size: int = 224
    max_images: int = 9
    cols: int = 3
    dpi: int = 220

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self.h1 = self.target_module.register_forward_hook(self._forward_hook)
        self.h2 = self.target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self) -> None:
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=False)

        acts = self.activations
        grads = self.gradients

        if acts is None or grads is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)

        cam = cam.squeeze(1)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

def load_image_tensor(path: Path, image_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im.resize((image_size, image_size)))
        x = tf(im)
    return x, arr

def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam = np.clip(cam, 0.0, 1.0)
    cmap = plt.get_cmap("jet")
    heat = cmap(cam)[:, :, :3]
    heat = (heat * 255).astype(np.uint8)
    blended = (rgb * (1.0 - alpha) + heat * alpha).astype(np.uint8)
    return blended

def find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "layer4"):
        layer4 = getattr(model, "layer4")
        if hasattr(layer4, "__len__") and len(layer4) > 0:
            return layer4[-1]
        return layer4
    raise RuntimeError("Could not locate target layer for GradCAM")

def draw_grid(images: List[np.ndarray], titles: List[str], out_path: Path, cols: int, dpi: int) -> None:
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

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_csv = Path("datasets/processed/index.csv")
    labels, _, idx_to_label = build_label_maps(index_csv)

    ckpt = Path("models/checkpoints/best_model.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model_cfg = ModelConfig(
        model_name="resnet18",
        num_classes=len(labels),
        pretrained=False,
        dropout=0.2,
    )
    model = build_model(model_cfg)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    target_layer = find_target_layer(model)
    cam_runner = GradCAM(model, target_layer)

    preds_csv = Path("outputs/phase3/test_predictions.csv")
    test_csv = Path("datasets/splits/test.csv")
    if not preds_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Missing outputs/phase3/test_predictions.csv or datasets/splits/test.csv")

    preds = pd.read_csv(preds_csv)
    test_df = pd.read_csv(test_csv)

    if len(preds) != len(test_df):
        raise RuntimeError("test.csv and test_predictions.csv length mismatch")

    df = test_df.copy()
    df["true_label"] = preds["true_label"].astype(str)
    df["pred_label"] = preds["pred_label"].astype(str)
    df["confidence"] = preds["confidence"].astype(float)
    df["pred_idx"] = preds["pred_idx"].astype(int)

    df = df.sample(frac=1.0, random_state=42).head(GradCAMConfig().max_images).reset_index(drop=True)

    cfg = GradCAMConfig()
    out_dir = Path("outputs/phase4")
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_images: List[np.ndarray] = []
    grid_titles: List[str] = []

    for i, r in df.iterrows():
        img_path = Path(r["path"])
        x, rgb = load_image_tensor(img_path, cfg.image_size)
        x = x.unsqueeze(0).to(device)

        class_idx = int(r["pred_idx"])
        cam = cam_runner(x, class_idx=class_idx).squeeze(0).detach().cpu().numpy()
        blended = overlay_cam(rgb, cam, alpha=0.45)

        pct = int(round(float(r["confidence"]) * 100))
        title = f"True: {r['true_label']} | Pred: {r['pred_label']} ({pct}%)"
        grid_images.append(blended)
        grid_titles.append(title)

        Image.fromarray(blended).save(out_dir / f"gradcam_{i+1:02d}.png")

    draw_grid(
        images=grid_images,
        titles=grid_titles,
        out_path=out_dir / "gradcam_gallery.png",
        cols=cfg.cols,
        dpi=cfg.dpi,
    )

    cam_runner.close()

    print("\nGRADCAM COMPLETE")
    print(f"Saved: {out_dir / 'gradcam_gallery.png'}")
    print(f"Saved individual overlays: {out_dir}\n")

if __name__ == "__main__":
    main()
