import argparse
from pathlib import Path

from src.data.split_dataset import make_splits
from src.data.validate_and_index import run as run_validate_and_index

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="satellite-landuse-classification",
        description="CLI for satellite land use classification pipeline.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path("datasets") / "raw"),
        help="Path to raw dataset folder containing class subfolders.",
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default=str(Path("src") / "config" / "class_mapping.yaml"),
        help="Path to class mapping config YAML.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path("datasets")),
        help="Root folder where processed and report outputs will be saved.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--sample-per-raw-class",
        type=int,
        default=10,
        help="How many images to sample per raw class for open/verify checks.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )

    parser.add_argument(
        "--phase",
        type=str,
        default="help",
        choices=["help", "data_check", "index", "split"],
        help="Which phase command to run.",
    )

    return parser

def phase_data_check(data_dir: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found inside: {data_dir}")

    print("\nDATA CHECK")
    print(f"Data dir: {data_dir.resolve()}")
    print(f"Found {len(class_dirs)} class folders:\n")

    for d in class_dirs:
        img_count = sum(1 for _ in d.glob("*"))
        print(f"  {d.name}: {img_count} files")

    print("")

def phase_index(data_dir: Path, config_path: Path, output_root: Path, seed: int, sample_per_raw_class: int) -> None:
    run_validate_and_index(
        data_dir=data_dir,
        config_path=config_path,
        out_root=output_root,
        seed=seed,
        sample_per_raw_class=sample_per_raw_class,
    )

def phase_split(output_root: Path, seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    index_csv = output_root / "processed" / "index.csv"
    out_dir = output_root / "splits"
    make_splits(
        index_csv=index_csv,
        out_dir=out_dir,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config_path = Path(args.config_path)
    output_root = Path(args.output_root)

    if args.phase == "help":
        parser.print_help()
        return

    if args.phase == "data_check":
        phase_data_check(data_dir)
        return

    if args.phase == "index":
        phase_index(
            data_dir=data_dir,
            config_path=config_path,
            output_root=output_root,
            seed=int(args.seed),
            sample_per_raw_class=int(args.sample_per_raw_class),
        )
        return

    if args.phase == "split":
        phase_split(
            output_root=output_root,
            seed=int(args.seed),
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
        )
        return

    raise RuntimeError(f"Unhandled phase: {args.phase}")

if __name__ == "__main__":
    main()
