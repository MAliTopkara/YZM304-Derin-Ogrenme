"""IMRAD_slides.pptx → PDF + per-slide PNG'ler (Microsoft PowerPoint COM otomasyonu).

Çıktılar:
    presentation/IMRAD_slides.pdf
    presentation/slides_png/slide_01.png ... slide_15.png

Çalıştırma (PowerPoint Windows'ta kurulu olmalı):
    python presentation/export_pptx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import win32com.client
from pywintypes import com_error

ROOT = Path(__file__).resolve().parent.parent
PPTX = ROOT / "presentation" / "IMRAD_slides.pptx"
PDF = ROOT / "presentation" / "IMRAD_slides.pdf"
PNG_DIR = ROOT / "presentation" / "slides_png"

# PowerPoint export format constants
PP_SAVE_AS_PDF = 32
PP_EXPORT_PNG = 18  # ppShapeFormatPNG, but for slide export we use PNG via Export()


def main() -> None:
    if not PPTX.exists():
        print(f"[ERROR] PPTX bulunamadı: {PPTX}")
        sys.exit(1)

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    # eski PNG'leri sil
    for old in PNG_DIR.glob("slide_*.png"):
        old.unlink()

    print(f"PowerPoint başlatılıyor...")
    pp = win32com.client.Dispatch("PowerPoint.Application")
    # Note: PowerPoint requires Visible=True on some Office versions; True=1, False=0
    # Setting Visible=False can fail; use True with WindowState minimized as workaround.
    pres = pp.Presentations.Open(str(PPTX), WithWindow=False)

    try:
        # 1) PDF export
        print(f"PDF üretiliyor: {PDF}")
        pres.SaveAs(str(PDF), PP_SAVE_AS_PDF)

        # 2) Her slaytı PNG olarak (1920x1080)
        n = pres.Slides.Count
        print(f"{n} slayt için PNG üretiliyor...")
        for i in range(1, n + 1):
            slide = pres.Slides(i)
            out = PNG_DIR / f"slide_{i:02d}.png"
            slide.Export(str(out), "PNG", 1920, 1080)
            print(f"  -> {out.name}")

    except com_error as e:
        print(f"[ERROR] COM error: {e}")
        sys.exit(1)
    finally:
        pres.Close()
        pp.Quit()
        print("PowerPoint kapatıldı.")

    # özet
    pdf_size = PDF.stat().st_size if PDF.exists() else 0
    png_count = len(list(PNG_DIR.glob("slide_*.png")))
    total_png = sum(p.stat().st_size for p in PNG_DIR.glob("slide_*.png"))
    print(f"\n{'='*60}")
    print(f"PDF:  {PDF} ({pdf_size/1024:.0f} KB)")
    print(f"PNG:  {PNG_DIR} ({png_count} dosya, toplam {total_png/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
