from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

@dataclass(frozen=True)
class MappingConfig:
    final_classes: List[str]
    mapping: Dict[str, str]


def load_mapping_config(path: Path) -> MappingConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    final_classes = data.get("final_classes")
    mapping = data.get("mapping")

    if not isinstance(final_classes, list) or not all(isinstance(x, str) for x in final_classes):
        raise ValueError("config must contain 'final_classes' as a list of strings")

    if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        raise ValueError("config must contain 'mapping' as a dict of string->string")

    final_set = set(final_classes)
    unknown_targets = sorted({v for v in mapping.values() if v not in final_set})
    if unknown_targets:
        raise ValueError(f"mapping contains targets not in final_classes: {unknown_targets}")

    return MappingConfig(final_classes=final_classes, mapping=mapping)


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VALID_EXTS


def verify_image(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def build_index(data_dir: Path, mapping_cfg: MappingConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    rows: List[dict] = []
    skipped: List[dict] = []

    for class_dir in class_dirs:
        raw_class = class_dir.name
        target_class = mapping_cfg.mapping.get(raw_class)

        files = sorted([p for p in class_dir.iterdir() if is_image_file(p)])

        if target_class is None:
            for p in files:
                skipped.append({"path": str(p.as_posix()), "raw_class": raw_class, "reason": "unmapped"})
            continue

        for p in files:
            rows.append({"path": str(p.as_posix()), "raw_class": raw_class, "label": target_class})

    df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)

    if df.empty:
        raise RuntimeError("No images indexed. Check datasets/raw, extensions, and src/config/class_mapping.yaml.")

    df["label"] = pd.Categorical(df["label"], categories=mapping_cfg.final_classes, ordered=True)
    df = df.sort_values(["label", "raw_class", "path"]).reset_index(drop=True)
    return df, skipped_df


def validate_images(df: pd.DataFrame, sample_per_raw_class: int, seed: int) -> pd.DataFrame:
    shuffled_idx = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).values.tolist()
    raw_classes = df["raw_class"].unique().tolist()

    checks: List[dict] = []
    seen: Dict[str, int] = {rc: 0 for rc in raw_classes}

    for idx in shuffled_idx:
        row = df.iloc[idx]
        rc = str(row["raw_class"])

        if seen[rc] >= sample_per_raw_class:
            continue

        ok = verify_image(Path(row["path"]))
        checks.append({"path": row["path"], "raw_class": rc, "label": str(row["label"]), "ok": bool(ok)})
        seen[rc] += 1

        if all(seen[x] >= sample_per_raw_class for x in raw_classes):
            break

    return pd.DataFrame(checks)


def run(
    data_dir: Path,
    config_path: Path,
    out_root: Path,
    seed: int = 42,
    sample_per_raw_class: int = 10,
) -> None:
    data_dir = data_dir.resolve()
    config_path = config_path.resolve()
    out_root = out_root.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    mapping_cfg = load_mapping_config(config_path)
    df, skipped_df = build_index(data_dir, mapping_cfg)

    check_df = validate_images(df, sample_per_raw_class=sample_per_raw_class, seed=seed)
    bad_df = check_df[check_df["ok"] == False] if not check_df.empty else pd.DataFrame()

    processed_dir = out_root / "processed"
    reports_dir = out_root / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    index_path = processed_dir / "index.csv"
    df.to_csv(index_path, index=False)

    if not skipped_df.empty:
        skipped_df.to_csv(reports_dir / "skipped_unmapped.csv", index=False)

    if not check_df.empty:
        check_df.to_csv(reports_dir / "image_open_checks.csv", index=False)

    label_counts = df["label"].value_counts(dropna=False).sort_index()
    raw_counts = (
        df.groupby(["raw_class", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["label", "raw_class"])
    )

    print("\nINDEX BUILD COMPLETE")
    print(f"Data dir: {data_dir}")
    print(f"Config:  {config_path}")
    print(f"Saved:   {index_path}")

    print("\nFinal class counts:")
    for k, v in label_counts.items():
        print(f"  {k}: {int(v)}")

    print("\nRaw -> Final mapping counts:")
    for _, r in raw_counts.iterrows():
        print(f"  {r['raw_class']} -> {r['label']}: {int(r['count'])}")

    if not bad_df.empty:
        print("\nWARNING: Some images failed to open during sampling.")
        print(f"Failed samples: {len(bad_df)}")
        print(f"See: {reports_dir / 'image_open_checks.csv'}\n")
    else:
        print("\nImage sampling check passed.\n")


if __name__ == "__main__":
    run(
        data_dir=Path("datasets/raw"),
        config_path=Path("src/config/class_mapping.yaml"),
        out_root=Path("datasets"),
        seed=42,
        sample_per_raw_class=10,
    )
