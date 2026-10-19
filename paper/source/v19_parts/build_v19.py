"""Build manuscript v19 (the short, hyperspectral-only rewrite of v18) from its parts.

    python paper/source/v19_parts/build_v19.py

1. Concatenates p*.md in name order, numbers citations ([@key]) by first appearance
   (IEEE), tables ({{T:key}}) in Roman and figures ({{F:key}}) in Arabic by first
   appearance, and appends the reference list ->
   paper/draft/MedMamba-SS-TRM_manuscript_v19.md
2. Converts that markdown to IEEEtran LaTeX -> paper/v19/main.tex, and writes every
   mermaid diagram to paper/v19/build/<name>.mmd for rendering to
   paper/v19/figures/<name>.pdf (see paper/v19/render_mermaid.sh).

Edit the p*.md parts, never the outputs. The reference texts are read from
paper/source/v18_parts/build_v18.py (REFS), so both versions cite identical entries.

LaTeX layout hints live in HTML comments just before a figure or table, invisible in
rendered markdown:  <!-- tex: wide; width=0.9; cols=lrr; x=Xrr; mermaid=name -->
"""
import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MD_OUT = ROOT / "paper/draft/MedMamba-SS-TRM_manuscript_v19.md"
TEX_DIR = ROOT / "paper/v19"


def load_v18_refs():
    src = (HERE.parent / "v18_parts/build_v18.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "REFS" for t in node.targets):
            return ast.literal_eval(node.value)
    sys.exit("REFS not found in build_v18.py")


REFS = load_v18_refs()
REFS["transformer"] = ('A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and '
                       'I. Polosukhin, "Attention is all you need," in *Proc. Adv. Neural Inf. Process. Syst. '
                       '(NeurIPS)*, vol. 30, 2017, pp. 5998–6008. [Online]. Available: '
                       'https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html')
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]


# ---------------------------------------------------------------------------- markdown
def build_markdown():
    text = "\n".join(p.read_text().rstrip() + "\n" for p in sorted(HERE.glob("p*.md")))
    cite = re.compile(r"\[@(\w+)\]")
    order = {}
    for m in cite.finditer(text):
        order.setdefault(m.group(1), len(order) + 1)
    missing = [k for k in order if k not in REFS]
    if missing:
        sys.exit(f"undefined citation keys: {missing}")
    text = cite.sub(lambda m: f"[{order[m.group(1)]}]", text)

    for tag, fmt, anchor in (("T", lambda i: ROMAN[i], r"\*\*TABLE \{\{T:(\w+)\}\}\*\*"),
                             ("F", lambda i: str(i + 1), r"^\*Fig\. \{\{F:(\w+)\}\}\."),):
        pat = re.compile(r"\{\{" + tag + r":(\w+)\}\}")
        seen = {}
        for m in pat.finditer(text):
            seen.setdefault(m.group(1), len(seen))
        placed = re.findall(anchor, text, flags=re.M)
        if placed != list(seen):     # LaTeX numbers floats in placement order
            sys.exit(f"{tag}: placement order {placed} differs from first-mention order {list(seen)}")
        text = pat.sub(lambda m: fmt(seen[m.group(1)]), text)
        print(f"{tag}: {len(seen)} ->", ", ".join(seen))

    leftover = re.findall(r"\{\{[^}]*\}\}|\[@[^\]]*\]", text)
    if leftover:
        sys.exit(f"unresolved placeholders: {leftover}")
    refs = "\n\n".join(f"[{n}] {REFS[k]}" for k, n in sorted(order.items(), key=lambda kv: kv[1]))
    md = text.rstrip() + "\n\n" + refs + "\n"
    MD_OUT.write_text(md)
    body = re.sub(r"```.*?```", "", text.split("## References")[0], flags=re.S)
    body = re.sub(r"<!--.*?-->", "", body)
    print(f"citations: {len(order)}; prose words (body, tables included): "
          f"{len(re.findall(r'[A-Za-z][A-Za-z-]*', body))}")
    print(f"[built] {MD_OUT}")
    return md


# ---------------------------------------------------------------------------- LaTeX
TEXT_UNICODE = {"−": "$-$", "×": "$\\times$", "±": "$\\pm$", "°": "$^{\\circ}$", "≥": "$\\geq$",
                "≤": "$\\leq$", "≈": "$\\approx$", "→": "$\\rightarrow$", "←": "$\\leftarrow$",
                "λ": "$\\lambda$", "η": "$\\eta$", "τ": "$\\tau$", "κ": "$\\kappa$", "\u00a0": "~"}


def inline(s):
    """Markdown inline -> LaTeX; $...$ math is protected and passed through."""
    maths = []

    def keep(m):
        maths.append(m.group(0))
        return f"\x00{len(maths) - 1}\x00"

    s = re.sub(r"\$[^$]+\$", keep, s)
    s = re.sub(r"([&%#_])", r"\\\1", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*\\])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r"\[(\d+)\]", r"\\cite{r\1}", s)
    for k, v in TEXT_UNICODE.items():
        s = s.replace(k, v)
    s = s.replace("$$", "")          # adjacent maps such as "6.2×" + "$..." never merge
    return re.sub("\x00(\\d+)\x00", lambda m: maths[int(m.group(1))], s)


def opts(comment):
    out = {}
    for part in comment.split(";"):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip() or True
    return out


def graphic(md_path, width, wide):
    p = (ROOT / "paper/draft" / md_path).resolve()
    pdf = p.with_suffix(".pdf")
    rel = Path("..") / (pdf if pdf.is_file() else p).relative_to(ROOT / "paper")
    return f"\\includegraphics[width={width}\\{'textwidth' if wide else 'linewidth'}]{{{rel.as_posix()}}}"


def table_tex(o, title, rows, note):
    wide = "wide" in o
    env = "table*" if wide else "table"
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, align, data = cells[0], cells[1], cells[2:]
    if "x" in o:
        spec = o["x"]
        begin = f"\\begin{{tabularx}}{{\\{'textwidth' if wide else 'columnwidth'}}}{{{spec}}}"
        end = "\\end{tabularx}"
    else:
        spec = o.get("cols") or "".join("r" if a.endswith(":") and not a.startswith(":") else
                                        "c" if a.startswith(":") else "l" for a in align)
        begin, end = f"\\begin{{tabular}}{{{spec}}}", "\\end{tabular}"
    line = lambda r: " & ".join(inline(c) for c in r) + " \\\\"
    out = [f"\\begin{{{env}}}[!t]", "\\centering", "\\footnotesize",
           "\\renewcommand{\\arraystretch}{1.12}", f"\\caption{{{inline(title)}}}", begin,
           "\\toprule", line(head), "\\midrule"]
    out += [line(r) for r in data]
    out += ["\\bottomrule", end]
    if note:
        out.append(f"\\par\\vspace{{2pt}}{{\\scriptsize\\raggedright {inline(note)}\\par}}")
    out.append(f"\\end{{{env}}}")
    return "\n".join(out)


def figure_tex(o, include, caption):
    wide = "wide" in o
    env = "figure*" if wide else "figure"
    return "\n".join([f"\\begin{{{env}}}[!t]", "\\centering", include,
                      f"\\caption{{{inline(caption)}}}", f"\\end{{{env}}}"])


def to_latex(md):
    body, refs = md.split("## References", 1)
    lines = body.split("\n")
    out, i, pending, mermaid = [], 0, {}, {}
    title = author = abstract = keywords = ""

    def caption_after(j):
        while j < len(lines) and not lines[j].strip():
            j += 1
        m = re.match(r"^\*Fig\. \d+\. (.*)\*$", lines[j].strip())
        if not m:
            sys.exit(f"figure without caption near line {j}: {lines[j][:60]}")
        return m.group(1), j + 1

    while i < len(lines):
        ln = lines[i].rstrip()
        s = ln.strip()
        if not s or s == "---":
            i += 1
            continue
        if s.startswith("# "):
            title = inline(s[2:]); i += 1; continue
        if s.startswith("**") and not author and not s.startswith("**TABLE"):
            author = inline(s.strip("*")) + "\\\\" + inline(lines[i + 1].strip())
            i += 2; continue
        if s.startswith("***Abstract*—"):
            abstract = inline(s[len("***Abstract*—"):-2]); i += 1; continue
        if s.startswith("***Index Terms*—"):
            keywords = inline(s[len("***Index Terms*—"):-2]); i += 1; continue
        m = re.match(r"^<!-- tex: (.*) -->$", s)
        if m:
            pending = opts(m.group(1)); i += 1; continue
        m = re.match(r"^## (?:([IVX]+)\. )?(.*)$", s)
        if m:
            out.append(("\\section{%s}" if m.group(1) else "\\section*{%s}") % inline(m.group(2)))
            i += 1; continue
        m = re.match(r"^### [A-Z]\. (.*)$", s)
        if m:
            out.append("\\subsection{%s}" % inline(m.group(1))); i += 1; continue
        if s.startswith("```mermaid"):
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            name = pending.get("mermaid") or sys.exit("mermaid block without <!-- tex: mermaid=name -->")
            mermaid[name] = "\n".join(lines[i + 1:j]) + "\n"
            cap, i = caption_after(j + 1)
            w = pending.get("width", "1.0")
            inc = f"\\includegraphics[width={w}\\{'textwidth' if 'wide' in pending else 'linewidth'}]{{figures/{name}.pdf}}"
            out.append(figure_tex(pending, inc, cap)); pending = {}; continue
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)$", s)
        if m:
            cap, i = caption_after(i + 1)
            out.append(figure_tex(pending, graphic(m.group(1), pending.get("width", "1.0"), "wide" in pending), cap))
            pending = {}; continue
        if re.match(r"^\*\*TABLE [IVX]+\*\*$", s):
            ttl = lines[i + 1].strip().strip("*")
            j = i + 2
            while not lines[j].strip().startswith("|"):
                j += 1
            rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(lines[j]); j += 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            note = ""
            if lines[j].strip().startswith("*") and not lines[j].strip().startswith(("*Fig.", "**")):
                note = lines[j].strip()[1:-1]; j += 1
            out.append(table_tex(pending, ttl, rows, note)); pending = {}; i = j; continue
        if s.startswith("$$"):
            eq = s
            while not (eq.endswith("$$") and len(eq) > 2):
                i += 1; eq += " " + lines[i].strip()
            out.append("\\begin{equation}\n" + eq[2:-2].strip() + "\n\\end{equation}")
            i += 1; continue
        if s.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append("\\item " + inline(lines[i].strip()[2:])); i += 1
            out.append("\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("<!--", "$$", "```", "## ", "### ", "**TABLE", "- ", "![")):
            para.append(lines[i].strip().lstrip("> ").strip()); i += 1
        if not para:
            sys.exit(f"cannot parse line {i}: {s[:80]}")
        out.append(inline(" ".join(para)))

    bib = []
    for m in re.finditer(r"^\[(\d+)\] (.*)$", refs, flags=re.M):
        n, t = m.groups()
        url = re.search(r"Available: (\S+)$", t)
        t2 = t[:url.start()] + "Available: \x01" if url else t
        t2 = re.sub(r'"([^"]*)"', r"``\1''", inline(t2))
        if url:
            t2 = t2.replace("\x01", f"\\url{{{url.group(1)}}}")
        bib.append(f"\\bibitem{{r{n}}} {t2}")

    tex = PREAMBLE + f"""
\\title{{{title}}}
\\author{{{author}}}
\\begin{{document}}
\\maketitle
\\begin{{abstract}}
{abstract}
\\end{{abstract}}
\\begin{{IEEEkeywords}}
{keywords}
\\end{{IEEEkeywords}}

""" + "\n\n".join(out) + "\n\n\\begin{thebibliography}{99}\n" + "\n".join(bib) + \
        "\n\\end{thebibliography}\n\\end{document}\n"
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    (TEX_DIR / "main.tex").write_text(tex)
    bdir = TEX_DIR / "build"
    bdir.mkdir(exist_ok=True)
    for name, src in mermaid.items():
        (bdir / f"{name}.mmd").write_text(src)
    print(f"[built] {TEX_DIR / 'main.tex'}; mermaid sources: {', '.join(mermaid)}")


PREAMBLE = r"""% Generated by paper/source/v19_parts/build_v19.py from the v19 markdown; do not edit.
% Build (XeTeX, via the host):  flatpak-spawn --host tectonic -X compile --keep-logs <abs>/main.tex
\documentclass[journal]{IEEEtran}
\usepackage{amsmath}
\usepackage{fontspec}
\usepackage{unicode-math}
\setmainfont{texgyretermes}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
  ItalicFont=*-italic, BoldItalicFont=*-bolditalic]
\setmathfont{texgyretermes-math.otf}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{cite}
\usepackage{url}
\renewcommand{\tabularxcolumn}[1]{>{\raggedright\arraybackslash}p{#1}}
"""


if __name__ == "__main__":
    to_latex(build_markdown())
