"""EDA — dataset uygunluk kontrolü.

Çalıştırma:
    python scripts/run_eda.py

Üretir:
    results/figures/eda_class_distribution.png
    results/figures/eda_sample_grid.png
    results/figures/eda_dimensions.png
    results/eda_report.md
"""
from __future__ import annotations

import io
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLASS_NAMES,
    FIGURES_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
)

random.seed(42)


def collect_stats():
    counts = {}
    sizes = Counter()
    modes = Counter()
    corrupt = []
    per_class_files = defaultdict(list)

    for cls in CLASS_NAMES:
        cls_dir = RAW_DATA_DIR / cls
        if not cls_dir.exists():
            print(f"[WARN] missing class dir: {cls_dir}")
            counts[cls] = 0
            continue
        files = sorted(cls_dir.iterdir())
        counts[cls] = len(files)
        per_class_files[cls] = files

        for f in files:
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    sizes[im.size] += 1
                    modes[im.mode] += 1
            except (UnidentifiedImageError, OSError) as e:
                corrupt.append((str(f), repr(e)))

    return counts, sizes, modes, corrupt, per_class_files


def plot_class_distribution(counts: dict, out: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    classes = list(counts.keys())
    values = list(counts.values())
    bars = ax.bar(classes, values, color="#4C72B0")
    ax.set_ylabel("Görsel sayısı")
    ax.set_title("Sınıf başına görsel dağılımı")
    ax.set_ylim(0, max(values) * 1.15)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 10, str(v),
                ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_sample_grid(per_class_files: dict, out: Path, n_per_class: int = 3):
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 2.6, n_classes * 1.6))
    for r, cls in enumerate(CLASS_NAMES):
        files = per_class_files.get(cls, [])
        if not files:
            continue
        sample = random.sample(files, min(n_per_class, len(files)))
        for c, f in enumerate(sample):
            ax = axes[r, c] if n_per_class > 1 else axes[r]
            with Image.open(f) as im:
                ax.imshow(im.convert("RGB"))
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(cls, fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=40)
    plt.suptitle("Sınıf başına 3 örnek görsel", y=1.0, fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_dimensions(sizes: Counter, modes: Counter, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # sizes pie
    labels = [f"{w}x{h}" for (w, h) in sizes.keys()]
    axes[0].pie(sizes.values(), labels=labels, autopct="%1.1f%%",
                colors=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("Görsel boyut dağılımı")

    # modes bar
    items = sorted(modes.items())
    axes[1].bar([k for k, _ in items], [v for _, v in items],
                color="#55A868")
    axes[1].set_title("Renk modu dağılımı (RGB / RGBA)")
    axes[1].set_ylabel("Görsel sayısı")
    for i, (k, v) in enumerate(items):
        axes[1].text(i, v + 50, f"{v}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(counts, sizes, modes, corrupt, out: Path):
    total = sum(counts.values())
    rgba_count = modes.get("RGBA", 0)
    rgb_count = modes.get("RGB", 0)

    lines = []
    lines.append("# EDA Raporu — Gameplay Images Dataset\n")
    lines.append(f"**Toplam görsel:** {total}\n")
    lines.append(f"**Sınıf sayısı:** {len(counts)}\n")
    lines.append(f"**Bozuk dosya:** {len(corrupt)}\n")
    lines.append("\n## Sınıf Dağılımı\n")
    lines.append("| Sınıf | Görsel sayısı |")
    lines.append("|---|---|")
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")

    lines.append("\n## Görsel Boyutları\n")
    lines.append("| Boyut | Adet |")
    lines.append("|---|---|")
    for k, v in sizes.items():
        lines.append(f"| {k[0]}×{k[1]} | {v} |")

    lines.append("\n## Renk Modları\n")
    lines.append("| Mode | Adet |")
    lines.append("|---|---|")
    for k, v in modes.items():
        lines.append(f"| {k} | {v} |")

    lines.append("\n## Uygunluk Kararı\n")
    balanced = max(counts.values()) - min(counts.values()) <= 50
    enough = total >= 8000
    classes_ok = sum(1 for v in counts.values() if v >= 700) >= 8
    decision = "✅ PROJE UYGUN" if (balanced and enough and classes_ok and len(corrupt) < 0.02 * total) else "⚠️ İncele"
    lines.append(f"- Toplam ≥ 8000: **{enough}** ({total})")
    lines.append(f"- Tüm sınıflar ≥ 700: **{classes_ok}**")
    lines.append(f"- Sınıflar dengeli (≤50 fark): **{balanced}**")
    lines.append(f"- Bozuk dosya oranı: **{len(corrupt)/max(total,1):.4%}**")
    lines.append(f"\n### {decision}\n")

    if rgba_count:
        lines.append("\n> ⚠️ **Not:** Bazı görseller RGBA modunda. "
                     "Preprocessing pipeline'ında RGB'ye dönüştürülecek (`src/transforms.py` içinde `_to_rgb`).\n")

    if corrupt:
        lines.append("\n## Bozuk Dosyalar\n")
        for f, err in corrupt[:20]:
            lines.append(f"- `{f}` — {err}")
        if len(corrupt) > 20:
            lines.append(f"- ... ve {len(corrupt) - 20} dosya daha")

    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset path: {RAW_DATA_DIR}")
    print("Stats toplanıyor...")
    counts, sizes, modes, corrupt, per_class_files = collect_stats()

    print(f"Toplam görsel: {sum(counts.values())}")
    print(f"Sınıf başına dağılım: {counts}")
    print(f"Boyutlar: {dict(sizes)}")
    print(f"Modlar: {dict(modes)}")
    print(f"Bozuk dosya: {len(corrupt)}")

    print("\nGrafikler üretiliyor...")
    plot_class_distribution(counts, FIGURES_DIR / "eda_class_distribution.png")
    plot_sample_grid(per_class_files, FIGURES_DIR / "eda_sample_grid.png")
    plot_dimensions(sizes, modes, FIGURES_DIR / "eda_dimensions.png")

    report_path = RESULTS_DIR / "eda_report.md"
    write_report(counts, sizes, modes, corrupt, report_path)
    print(f"\n✅ EDA raporu: {report_path}")
    print(f"✅ Figürler: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
