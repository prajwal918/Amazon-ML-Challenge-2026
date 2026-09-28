#!/usr/bin/env python3
"""
Generates publication-quality PDF documents for Amazon ML Challenge 2026:
1. Amazon_ML_Challenge_2026_Methodology_Document.pdf (from Documentation_template.md)
2. Amazon_ML_Challenge_2026_Technical_Architecture.pdf (from documentation.md)
"""

import os
from pathlib import Path
import os, json
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(r".")
DESKTOP = Path(r"~/Desktop")

CSS = """
@page { size: A4; margin: 15mm; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #1e293b; font-size: 10.5pt; max-width: 850px; margin: 0 auto; }
h1 { color: #0f172a; border-bottom: 2px solid #cbd5e1; padding-bottom: 6px; font-size: 18pt; margin-top: 20px; }
h2 { color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; font-size: 14pt; margin-top: 18px; }
h3 { color: #334155; font-size: 12pt; margin-top: 14px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.5pt; }
th, td { border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }
th { background-color: #f1f5f9; font-weight: 600; color: #0f172a; }
tr:nth-child(even) { background-color: #f8fafc; }
code { background-color: #f1f5f9; padding: 2px 4px; border-radius: 3px; font-family: Consolas, monospace; font-size: 9.5pt; color: #0969da; }
pre { background-color: #0f172a; color: #f8fafc; padding: 10px; border-radius: 5px; font-size: 9pt; overflow-x: auto; }
pre code { background: none; color: inherit; padding: 0; }
blockquote { border-left: 4px solid #2563eb; margin: 8px 0; padding-left: 12px; color: #475569; font-style: italic; }
"""

def generate_pdf(src_md, out_pdf, title):
    if not src_md.exists():
        print(f"Error: {src_md} does not exist!")
        return False
    
    with open(src_md, "r", encoding="utf-8", errors="ignore") as f:
        md_text = f.read()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # Load marked.js to parse markdown inside chromium
        page.set_content(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
            <style>{CSS}</style>
        </head>
        <body>
            <div id="content"></div>
            <script>
                const raw = {json.dumps(md_text)};
                if (typeof marked !== 'undefined') {{
                    document.getElementById('content').innerHTML = marked.parse(raw);
                }} else {{
                    document.getElementById('content').innerText = raw;
                }}
            </script>
        </body>
        </html>
        """)
        page.wait_for_timeout(1000)
        page.pdf(
            path=str(out_pdf),
            format="A4",
            margin={"top": "15mm", "bottom": "15mm", "left": "15mm", "right": "15mm"},
            print_background=True
        )
        browser.close()
    
    size_kb = out_pdf.stat().st_size / 1024
    print(f"Successfully generated: {out_pdf.name} ({size_kb:.1f} KB)")
    return True

if __name__ == "__main__":
    print("Generating official Amazon ML Challenge 2026 Documentation PDFs...")
    # PDF 1: Methodology Document
    p1 = DESKTOP / "Amazon_ML_Challenge_2026_Methodology_Document.pdf"
    generate_pdf(BASE_DIR / "Documentation_template.md", p1, "Amazon ML Challenge 2026 - Methodology Document")

    # PDF 2: Technical Architecture Document
    p2 = DESKTOP / "Amazon_ML_Challenge_2026_Technical_Architecture.pdf"
    generate_pdf(BASE_DIR / "documentation.md", p2, "Amazon ML Challenge 2026 - Technical Architecture")

    print("\nAll 2 PDFs generated directly on Desktop!")
