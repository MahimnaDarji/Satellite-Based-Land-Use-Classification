from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn

@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "resnet18"
    num_classes: int = 4
    pretrained: bool = True
    dropout: float = 0.0


def build_model(cfg: ModelConfig) -> nn.Module:
    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
    )

    if cfg.dropout > 0.0:
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=cfg.dropout),
                nn.Linear(in_features, cfg.num_classes),
            )
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=cfg.dropout),
                nn.Linear(in_features, cfg.num_classes),
            )

    return model

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    cfg = ModelConfig(
        model_name="resnet18",
        num_classes=4,
        pretrained=True,
        dropout=0.2,
    )

    model = build_model(cfg)
    n_params = count_parameters(model)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print("\nMODEL SANITY CHECK")
    print(f"Model: {cfg.model_name}")
    print(f"Trainable params: {n_params}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print("")
