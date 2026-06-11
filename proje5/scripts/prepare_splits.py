"""Generate stratified train/val/test CSVs from FracAtlas folder structure.

Outputs (in outputs/splits/):
    train.csv, val.csv, test.csv  -- columns: image_path, label
    split_summary.json             -- counts per split/class
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    FRACTURED_DIR,
    NON_FRACTURED_DIR,
    SEED,
    SPLITS_DIR,
    TrainConfig,
)
from src.utils import save_json


def collect_samples() -> pd.DataFrame:
    rows = []
    for p in sorted(FRACTURED_DIR.glob("*.jpg")):
        rows.append({"image_path": str(p), "label": 1})
    for p in sorted(NON_FRACTURED_DIR.glob("*.jpg")):
        rows.append({"image_path": str(p), "label": 0})
    if not rows:
        raise RuntimeError(f"No images found under {FRACTURED_DIR} or {NON_FRACTURED_DIR}")
    return pd.DataFrame(rows)


def main() -> None:
    cfg = TrainConfig()
    df = collect_samples()
    print(f"Total images: {len(df)}")
    print(df["label"].value_counts().rename({0: "non_fractured", 1: "fractured"}))

    # First split off test, then split remainder into train/val, all stratified.
    trainval, test = train_test_split(
        df,
        test_size=cfg.test_split,
        stratify=df["label"],
        random_state=SEED,
    )
    val_relative = cfg.val_split / (1.0 - cfg.test_split)
    train, val = train_test_split(
        trainval,
        test_size=val_relative,
        stratify=trainval["label"],
        random_state=SEED,
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "val.csv", index=False)
    test.to_csv(SPLITS_DIR / "test.csv", index=False)

    summary = {
        "total": int(len(df)),
        "seed": SEED,
        "val_split": cfg.val_split,
        "test_split": cfg.test_split,
        "splits": {
            name: {
                "total": int(len(part)),
                "non_fractured": int((part["label"] == 0).sum()),
                "fractured": int((part["label"] == 1).sum()),
            }
            for name, part in [("train", train), ("val", val), ("test", test)]
        },
    }
    save_json(summary, SPLITS_DIR / "split_summary.json")
    print("\nSplit summary:")
    for name, counts in summary["splits"].items():
        print(f"  {name}: {counts}")
    print(f"\nSaved splits to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
