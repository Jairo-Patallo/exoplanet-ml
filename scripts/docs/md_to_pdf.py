"""Convierte un Markdown de docs/ en PDF con Chrome o Edge en modo headless.

Uso: python scripts/docs/md_to_pdf.py docs/documentacion.md [--compact]
Requiere el paquete `markdown` y Chrome o Edge instalado.
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]

# Las dos fórmulas del documento, en HTML (Chrome no renderiza LaTeX).
MATH = {
    r"$$\delta \approx \left(\frac{R_p}{R_\star}\right)^2$$":
        '<p class="math">δ ≈ (R<sub>p</sub> / R<sub>★</sub>)<sup>2</sup></p>',
    r"$R_\star\sqrt{\delta}$": "R<sub>★</sub>·√δ",
}

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.45; color: #1d1d1f; }
h1 { font-size: 22pt; color: #0b3d75; margin: 0 0 6pt; }
h2 { font-size: 15pt; color: #0b3d75; border-bottom: 2px solid #0b3d75;
     padding-bottom: 3pt; margin-top: 18pt; break-before: page; }
h3 { font-size: 12pt; color: #24578f; margin: 12pt 0 4pt; }
h2, h3 { break-after: avoid; }
a { color: #24578f; text-decoration: none; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; background: #eef3fa;
             border-left: 4px solid #2b6cb0; }
blockquote p { margin: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9.5pt;
        break-inside: avoid; }
th, td { border: 1px solid #c9d3df; padding: 3pt 6pt; text-align: left; vertical-align: top; }
th { background: #e3ebf5; }
tr:nth-child(even) td { background: #f7f9fc; }
img { display: block; max-width: 100%; max-height: 105mm; margin: 8pt auto 2pt; }
p:has(> img) { break-inside: avoid; break-after: avoid; margin-bottom: 0; }
p:has(> img) + p em { display: block; text-align: center; font-size: 9pt; color: #555; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f1f3f5; padding: 0 2pt; }
hr { display: none; }
.math { text-align: center; font-size: 13pt; }
"""

COMPACT_CSS = """
@page { size: A4; margin: 11mm 13mm; }
body { font-size: 9.4pt; line-height: 1.3; }
h1 { font-size: 16pt; margin-bottom: 2pt; }
h2 { font-size: 11.5pt; break-before: auto; margin: 7pt 0 3pt; padding-bottom: 1pt; }
p, ul { margin: 3pt 0; }
li { margin: 1pt 0; }
table { font-size: 8.6pt; margin: 3pt 0 5pt; }
th, td { padding: 1.5pt 4pt; }
img { max-height: 62mm; margin: 3pt auto 0; }
p:has(> img) + p em { font-size: 8pt; }
.fig2 { display: flex; gap: 6pt; align-items: flex-start; }
.fig2 > div { flex: 1; }
.fig2 img { max-height: 52mm; }
.fig2 em { display: block; text-align: center; font-size: 8pt; color: #555; }
"""


def build_html(md_path: Path, compact: bool) -> str:
    text = md_path.read_text(encoding="utf-8")
    for latex, html in MATH.items():
        text = text.replace(latex, html)
    text = re.sub(r"(?<![(<])(https?://[^\s)]+)", r"<\1>", text)  # enlaces sueltos
    body = markdown.markdown(
        text,
        extensions=["tables", "toc", "md_in_html", "sane_lists"],
        extension_configs={"toc": {"slugify": slugify_unicode}},
    )
    title = re.search(r"^# (.+)$", text, re.M).group(1)
    css = CSS + (COMPACT_CSS if compact else "")
    base = md_path.parent.resolve().as_uri() + "/"
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f'<base href="{base}"><title>{title}</title><style>{css}</style>'
            f"</head><body>{body}</body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md", type=Path)
    ap.add_argument("--compact", action="store_true", help="estilo denso para el resumen")
    args = ap.parse_args()

    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        raise SystemExit("No se encontró Chrome ni Edge")

    out = args.md.with_suffix(".pdf").resolve()
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "doc.html"
        html.write_text(build_html(args.md, args.compact), encoding="utf-8")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", f"--print-to-pdf={out}",
                        html.as_uri()], check=True, capture_output=True)
    print(f"PDF: {out}")


if __name__ == "__main__":
    main()
