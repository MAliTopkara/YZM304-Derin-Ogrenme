"""Stratified train/val/test split (70/15/15).

Dataset/<class>/*.png  →  data/processed/{train,val,test}/<class>/*.png

Strateji: dosyaları kopyalamak yerine **hardlink** (Windows'ta destekli) kullanır;
yer kazandırır, hız 10× artar. Hardlink başarısız olursa kopyalamaya düşer.

Çalıştırma:
    python scripts/run_split.py
    python scripts/run_split.py --copy   # hardlink yerine kopyala
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLASS_NAMES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SPLIT_RATIOS,
    SPLIT_SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)


def link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except (OSError, AttributeError):
        shutil.copy2(src, dst)


def split_one_class(cls: str, use_copy: bool) -> tuple[int, int, int]:
    files = sorted((RAW_DATA_DIR / cls).iterdir())
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS

    # iki adımlı stratified-yerine basit shuffle: tek sınıf, sklearn 'stratify' gereksiz
    train_files, temp_files = train_test_split(
        files, train_size=train_ratio, random_state=SPLIT_SEED, shuffle=True
    )
    rel_val = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(
        temp_files, train_size=rel_val, random_state=SPLIT_SEED, shuffle=True
    )

    for f in train_files:
        link_or_copy(f, TRAIN_DIR / cls / f.name, use_copy)
    for f in val_files:
        link_or_copy(f, VAL_DIR / cls / f.name, use_copy)
    for f in test_files:
        link_or_copy(f, TEST_DIR / cls / f.name, use_copy)

    return len(train_files), len(val_files), len(test_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true",
                    help="Hardlink yerine dosyaları kopyala (~2.5GB ek yer)")
    ap.add_argument("--clean", action="store_true",
                    help="Önce data/processed/ içeriğini temizle")
    args = ap.parse_args()

    if args.clean and PROCESSED_DATA_DIR.exists():
        print(f"[clean] {PROCESSED_DATA_DIR}")
        for sub in (TRAIN_DIR, VAL_DIR, TEST_DIR):
            if sub.exists():
                shutil.rmtree(sub)

    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    mode = "copy" if args.copy else "hardlink"
    print(f"Mode: {mode}")
    print(f"Ratios: train={SPLIT_RATIOS[0]}, val={SPLIT_RATIOS[1]}, test={SPLIT_RATIOS[2]}")
    print(f"Seed: {SPLIT_SEED}")
    print()

    totals = [0, 0, 0]
    for cls in CLASS_NAMES:
        tr, va, te = split_one_class(cls, use_copy=args.copy)
        totals[0] += tr
        totals[1] += va
        totals[2] += te
        print(f"  {cls:20s}  train={tr:4d}  val={va:4d}  test={te:4d}")

    print()
    print(f"Toplam: train={totals[0]}, val={totals[1]}, test={totals[2]}")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Val dir:   {VAL_DIR}")
    print(f"Test dir:  {TEST_DIR}")


if __name__ == "__main__":
    main()
