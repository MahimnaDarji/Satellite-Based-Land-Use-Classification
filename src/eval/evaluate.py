from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.training.datamodule import DataConfig, build_label_maps, build_loaders
from src.training.model import ModelConfig, build_model

def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    model_cfg = ModelConfig(
        model_name="resnet18",
        num_classes=num_classes,
        pretrained=False,
        dropout=0.2,
    )
    model = build_model(model_cfg)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def predict(
    model: torch.nn.Module, loader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_confs: List[float] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        conf = probs.max(dim=1).values

        all_targets.extend(y.cpu().numpy().tolist())
        all_preds.extend(pred.cpu().numpy().tolist())
        all_confs.extend(conf.cpu().numpy().tolist())

    return np.array(all_targets), np.array(all_preds), np.array(all_confs)

def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_csv = Path("datasets/processed/index.csv")
    labels, label_to_idx, idx_to_label = build_label_maps(index_csv)

    data_cfg = DataConfig()
    _, _, test_loader = build_loaders(data_cfg, label_to_idx)

    checkpoint_path = Path("models/checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(checkpoint_path, num_classes=len(labels), device=device)

    y_true, y_pred, y_conf = predict(model, test_loader, device=device)

    report = classification_report(
        y_true,
        y_pred,
        target_names=[idx_to_label[i] for i in range(len(labels))],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    out_dir = Path("outputs/phase3")
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "classification_report.csv"
    report_df.to_csv(report_path, index=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_path = out_dir / "confusion_matrix.png"
    save_confusion_matrix(cm, [idx_to_label[i] for i in range(len(labels))], cm_path)

    acc = float((y_true == y_pred).mean())

    preds_df = pd.DataFrame(
        {
            "true_idx": y_true,
            "pred_idx": y_pred,
            "true_label": [idx_to_label[i] for i in y_true],
            "pred_label": [idx_to_label[i] for i in y_pred],
            "confidence": y_conf,
        }
    )
    preds_path = out_dir / "test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    print("\nEVALUATION COMPLETE")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Saved: {report_path}")
    print(f"Saved: {cm_path}")
    print(f"Saved: {preds_path}\n")


if __name__ == "__main__":
    main()
