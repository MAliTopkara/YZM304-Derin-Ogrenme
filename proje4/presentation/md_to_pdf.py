"""Markdown → güzel HTML → PDF (Edge headless ile).

Çalıştırma:
    python presentation/md_to_pdf.py presentation/talk_script.md
    # → presentation/talk_script.pdf
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
ROOT = Path(__file__).resolve().parent.parent

# Slaytlarla uyumlu yeşil-beyaz CSS
CSS = """
@page {
  size: A4;
  margin: 22mm 18mm 22mm 18mm;
}

* { box-sizing: border-box; }

html, body {
  font-family: "Calibri", "Segoe UI", Roboto, Arial, sans-serif;
  font-size: 11.5pt;
  line-height: 1.55;
  color: #1f2937;
  margin: 0;
  padding: 0;
}

h1 {
  font-size: 22pt;
  color: #14532d;
  border-bottom: 3px solid #16a34a;
  padding-bottom: 6px;
  margin-top: 0;
  margin-bottom: 16pt;
  page-break-after: avoid;
}

h2 {
  font-size: 16pt;
  color: #14532d;
  margin-top: 22pt;
  margin-bottom: 8pt;
  padding: 6px 10px;
  background: #dcfce7;
  border-left: 4px solid #16a34a;
  page-break-after: avoid;
}

h3 {
  font-size: 13pt;
  color: #166534;
  margin-top: 14pt;
  margin-bottom: 6pt;
  page-break-after: avoid;
}

p { margin: 6pt 0; }

strong { color: #14532d; }

em { color: #475569; font-style: italic; }

blockquote {
  border-left: 3px solid #16a34a;
  background: #f0fdf4;
  padding: 8pt 14pt;
  margin: 10pt 0;
  color: #1e293b;
  font-size: 11pt;
  border-radius: 4px;
  page-break-inside: avoid;
}

blockquote p {
  margin: 4pt 0;
}

ul, ol {
  margin: 6pt 0 6pt 8pt;
  padding-left: 18pt;
}

li {
  margin: 3pt 0;
}

li::marker {
  color: #16a34a;
}

hr {
  border: none;
  border-top: 1px solid #cbd5e1;
  margin: 18pt 0;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 10pt 0;
  font-size: 10pt;
  page-break-inside: avoid;
}

th {
  background: #14532d;
  color: white;
  padding: 6pt 8pt;
  text-align: left;
  font-weight: 600;
}

td {
  padding: 5pt 8pt;
  border: 1px solid #cbd5e1;
}

tr:nth-child(even) td {
  background: #f0fdf4;
}

code {
  font-family: Consolas, "Courier New", monospace;
  font-size: 10pt;
  background: #f1f5f9;
  padding: 1pt 4pt;
  border-radius: 3px;
  color: #0f172a;
}

pre {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-left: 3px solid #16a34a;
  padding: 8pt;
  font-family: Consolas, monospace;
  font-size: 10pt;
  overflow-x: auto;
  page-break-inside: avoid;
}

a { color: #16a34a; text-decoration: none; }
a:hover { text-decoration: underline; }

/* başlık çıkartmaları */
h1::before {
  content: "";
}

/* ilk slayt başlığı vurgu */
.title-banner {
  background: linear-gradient(135deg, #14532d 0%, #16a34a 100%);
  color: white;
  padding: 18pt 16pt;
  border-radius: 8pt;
  margin-bottom: 18pt;
  text-align: center;
}
.title-banner h1 {
  color: white;
  border-bottom: none;
  padding: 0;
  margin: 0 0 4pt 0;
}
.title-banner .meta {
  font-size: 10pt;
  opacity: 0.9;
}

/* sayfa altı */
.footer-info {
  font-size: 9pt;
  color: #64748b;
  text-align: center;
  margin-top: 30pt;
  border-top: 1px solid #e2e8f0;
  padding-top: 8pt;
}
"""


def md_to_html(md_path: Path, css: str, title: str) -> str:
    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <div class="title-banner">
    <div class="meta">YZM304 — Derin Öğrenme · Proje 4</div>
    <div class="meta" style="margin-top:4pt">Mehmet Ali Topkara · 23291093</div>
  </div>
  {html_body}
  <div class="footer-info">
    GitHub: github.com/MAliTopkara/YZM304-Derin-Ogrenme/tree/main/proje4
  </div>
</body>
</html>
"""


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    if not EDGE.exists():
        print(f"[ERROR] Edge bulunamadı: {EDGE}")
        sys.exit(1)

    cmd = [
        str(EDGE),
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        # url file://... mutlak path
        html_path.absolute().as_uri(),
    ]
    print(f"Edge'e PDF için çağrı yapılıyor...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"[ERROR] Edge return code: {result.returncode}")
        print(result.stderr[:500] if result.stderr else "")
        sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python presentation/md_to_pdf.py <input.md> [output.pdf]")
        sys.exit(1)

    md_path = Path(sys.argv[1]).resolve()
    if not md_path.exists():
        print(f"[ERROR] MD dosyası yok: {md_path}")
        sys.exit(1)

    pdf_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else md_path.with_suffix(".pdf")
    title = md_path.stem.replace("_", " ").title()

    # geçici html dosyası
    tmp_dir = Path(tempfile.mkdtemp(prefix="md2pdf_"))
    html_path = tmp_dir / "doc.html"
    html_text = md_to_html(md_path, CSS, title)
    html_path.write_text(html_text, encoding="utf-8")
    print(f"HTML üretildi: {html_path}")

    try:
        html_to_pdf(html_path, pdf_path)
        print(f"\nPDF: {pdf_path}")
        print(f"Boyut: {pdf_path.stat().st_size / 1024:.0f} KB")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
