#!/usr/bin/env python
"""One-command rebuild of the v11 manuscript from its parts.

    python paper/source/v11_parts/rebuild.py [output.md]

1. restore the unfilled conclusion template (q5_unfilled.md -> q5.md)
2. fill_v11.py: every data-dependent table/paragraph from paper/source/v19_analysis.json
   (writes q4a_filled.md, q4h_filled.md, q5_filled.md, abstract.md)
3. assemble Section VI in order A-C | D-G | H | I | J-K | L into q4.md
4. q5_filled.md -> q5.md, then build_v11.py (citations, tables, figures numbered)

Edit q1/q2/q3/q4a/q4b/q4c/q4h/q4l/q5_unfilled, never the *_filled files or q4.md/q5.md,
which are regenerated. Default output: paper/draft/MedMamba-SS-TRM_manuscript_v11.md
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper/draft/MedMamba-SS-TRM_manuscript_v11.md"

shutil.copy(HERE / "q5_unfilled.md", HERE / "q5.md")
subprocess.run([sys.executable, str(HERE / "fill_v11.py")], check=True, cwd=HERE, stdout=subprocess.DEVNULL)
b = (HERE / "q4b_filled.md").read_text()
i = b.index("### J. Reconstruction")
parts = [(HERE / "q4a_filled.md").read_text(), b[:i], (HERE / "q4h_filled.md").read_text(),
         (HERE / "q4c.md").read_text(), b[i:], (HERE / "q4l.md").read_text()]
(HERE / "q4.md").write_text("\n\n".join(p.strip() for p in parts) + "\n\n")
shutil.copy(HERE / "q5_filled.md", HERE / "q5.md")
subprocess.run([sys.executable, str(HERE / "build_v11.py"), str(out)], check=True, cwd=HERE)
print(f"[rebuilt] {out}")
