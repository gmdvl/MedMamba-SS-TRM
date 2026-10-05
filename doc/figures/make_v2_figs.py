#!/usr/bin/env python3
"""The fourteen figures added for the v2 deck, in the existing visual language.

Same palette and type scale as fig01-fig20 (see README.md), so the two
generations of figure sit together without a style clash. Every figure asserts
that nothing it drew escapes the canvas - a diagram that overflows on a
projector is worse than no diagram.

Run:  python3 make_v2_figs.py
"""
import os

OUT = os.path.dirname(os.path.abspath(__file__))

INK, MUTED, GREY = '#1c2430', '#7b8794', '#5a6472'
ORANGE, GREEN, PURPLE, RED = '#c2703a', '#4e8c6a', '#6a5a92', '#b4553f'
ORANGE_F, GREEN_F, PURPLE_F, RED_F = '#fbeee3', '#e8f3ed', '#ece9f3', '#f8e7e2'
PANEL, WHITE = '#f2f4f7', '#ffffff'
BAR_GREY = '#dbe0e6'   # a grey bar has to read against the PANEL track behind it

STYLE = """<style>
  .t   {font-size:15px;fill:#1c2430}
  .tb  {font-size:15.5px;font-weight:600;fill:#1c2430}
  .ts  {font-size:12.5px;fill:#7b8794}
  .tsb {font-size:12.5px;font-weight:600;fill:#7b8794}
  .ti  {font-size:21px;font-weight:600;fill:#1c2430}
  .tsub{font-size:14px;fill:#7b8794}
  .big {font-size:30px;font-weight:600;fill:#1c2430}
  .huge{font-size:38px;font-weight:600;fill:#1c2430}
  .mono{font-family:"SFMono-Regular",Consolas,monospace;font-size:13.5px;fill:#1c2430}
  .monb{font-family:"SFMono-Regular",Consolas,monospace;font-size:13.5px;font-weight:600;fill:#1c2430}
</style>"""

MARKERS = [('ar', GREY), ('aro', ORANGE), ('arg', GREEN), ('arp', PURPLE), ('arr', RED)]


class Fig:
    """A canvas that records what it drew, so it can check its own bounds."""

    def __init__(self, name, w=1040, h=470):
        self.name, self.w, self.h = name, w, h
        self.parts = []
        self.bounds = []          # (x0, y0, x1, y1) of every drawn element
        self.texts = []           # the same, for text only, plus the string

    # -- bookkeeping ------------------------------------------------------
    def _bb(self, x0, y0, x1, y1):
        self.bounds.append((x0, y0, x1, y1))

    def raw(self, s):
        self.parts.append(s)

    # -- primitives -------------------------------------------------------
    def box(self, x, y, w, h, fill=WHITE, stroke=GREY, rx=7, sw=1.6, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>')
        self._bb(x, y, x + w, y + h)
        return (x, y, w, h)

    def text(self, x, y, s, cls='t', anchor='start', fill=None, size=None, weight=None):
        # text-anchor MUST be inline style: the CSS class wins over the attribute.
        st = [f'text-anchor:{anchor}']
        if fill:
            st.append(f'fill:{fill}')
        if size:
            st.append(f'font-size:{size}px')
        if weight:
            st.append(f'font-weight:{weight}')
        self.raw(f'<text x="{x}" y="{y}" class="{cls}" style="{";".join(st)}">{esc(s)}</text>')
        fs = size or {'ts': 12.5, 'tsb': 12.5, 'ti': 21, 'tsub': 14, 'big': 30,
                      'huge': 38, 'mono': 13.5, 'monb': 13.5}.get(cls, 15)
        w = tw(s, fs, weight == 600 or cls in ('tb', 'monb', 'ti', 'big', 'huge', 'tsb'),
               cls in ('mono', 'monb'))
        x0 = x if anchor == 'start' else (x - w / 2 if anchor == 'middle' else x - w)
        self._bb(x0, y - fs, x0 + w, y + fs * 0.3)
        # tighter box for the collision check: real glyphs, not line box
        self.texts.append((x0, y - fs * 0.74, x0 + w, y + fs * 0.20, s))

    def line(self, x1, y1, x2, y2, color=GREY, sw=1.8, arrow=True, dash=None):
        mk = dict(MARKERS)
        mid = next(k for k, v in MARKERS if v == color) if color in mk.values() else 'ar'
        a = f' marker-end="url(#{mid})"' if arrow else ''
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.raw(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                 f'stroke-width="{sw}" fill="none"{a}{d}/>')
        self._bb(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def path(self, d, color=GREY, sw=1.8, arrow=True, fill='none', dash=None):
        mk = next((k for k, v in MARKERS if v == color), 'ar')
        a = f' marker-end="url(#{mk})"' if arrow else ''
        da = f' stroke-dasharray="{dash}"' if dash else ''
        self.raw(f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="{fill}"{a}{da}/>')

    def circle(self, cx, cy, r, fill=WHITE, stroke=GREY, sw=1.6):
        self.raw(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}"/>')
        self._bb(cx - r, cy - r, cx + r, cy + r)

    # -- compounds --------------------------------------------------------
    def head(self, title, sub=None):
        self.text(30, 38, title, 'ti')
        if sub:
            self.text(30, 60, sub, 'tsub')

    def badge(self, cx, cy, n, color):
        """A numbered callout circle."""
        self.circle(cx, cy, 12, fill=color, stroke=color)
        self.text(cx, cy + 4.5, str(n), 'tb', 'middle', fill=WHITE, size=13)

    def card(self, x, y, w, h, heading, lines, fill=WHITE, stroke=GREY, head_size=15.5):
        """A titled box whose body lines are centred."""
        self.box(x, y, w, h, fill, stroke)
        self.text(x + w / 2, y + 26, heading, 'tb', 'middle', size=head_size)
        yy = y + 48
        for ln, cls in lines:
            self.text(x + w / 2, yy, ln, cls, 'middle')
            yy += 21 if cls != 'ts' else 18
        return yy

    def banner(self, x, y, w, s, fill=PANEL, stroke=GREY, h=44, cls='tb', color=None):
        self.box(x, y, w, h, fill, stroke)
        self.text(x + w / 2, y + h / 2 + 5.5, s, cls, 'middle', fill=color)
        return y + h

    def hbar(self, x, y, w, h, frac, fill, stroke, label=None, value=None, lw=170):
        """A horizontal bar with a right-hand label; track drawn behind it."""
        if label is not None:
            self.text(x - 10, y + h / 2 + 5, label, 'ts', 'end')
        self.box(x, y, w, h, PANEL, '#e3e6ea', rx=4, sw=1)
        bw = max(3.0, w * frac)
        self.box(x, y, bw, h, fill, stroke, rx=4, sw=1.4)
        if value is not None:
            self.text(x + w + 12, y + h / 2 + 5, value, 'tb', 'start')

    # -- output -----------------------------------------------------------
    def save(self):
        # Two labels sharing pixels is always a bug, and it is the one kind of
        # damage that a bounds check on the canvas cannot see.
        for i, a in enumerate(self.texts):
            for b in self.texts[i + 1:]:
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox > 1.5 and oy > 1.5:
                    raise AssertionError(
                        f'{self.name}: text collision \u2014 {a[4]!r} overlaps {b[4]!r} '
                        f'by {ox:.0f}x{oy:.0f}px')
        for (x0, y0, x1, y1) in self.bounds:
            assert -1 <= x0 and x1 <= self.w + 1, f'{self.name}: x overflow {x0:.0f}..{x1:.0f} of {self.w}'
            assert -1 <= y0 and y1 <= self.h + 1, f'{self.name}: y overflow {y0:.0f}..{y1:.0f} of {self.h}'
        defs = ''.join(
            f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for i, c in MARKERS)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
               f'width="{self.w}" height="{self.h}" '
               f'font-family="Helvetica Neue, Helvetica, Arial, sans-serif">\n'
               f'<defs>{defs}{STYLE}</defs>\n'
               f'<rect width="{self.w}" height="{self.h}" fill="{WHITE}"/>\n'
               + '\n'.join(self.parts) + '\n</svg>\n')
        p = os.path.join(OUT, self.name)
        open(p, 'w').write(svg)
        print(f'  {self.name}  {self.w}x{self.h}')


def esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


# Helvetica advance widths, normalised to font size. Good to a few percent,
# which is all the bounds check needs.
_NARROW = set('iljtfrI.,:;\'`|!()[]{} ')
_WIDE = set('MWmw@%')

def tw(s, fs, bold=False, mono=False):
    if mono:
        return len(str(s)) * fs * 0.60
    t = 0.0
    for ch in str(s):
        t += 0.30 if ch in _NARROW else (0.83 if ch in _WIDE else 0.545)
    return t * fs * (1.045 if bold else 1.0)


# =========================================================== 21 · the spine
def fig21():
    f = Fig('fig21_story_map.svg', 1040, 430)
    f.head('Two numbers are frozen into MedMamba’s weights',
           'Both are decided at build time. Both cost you something real. This talk deletes them.')

    f.text(120, 106, 'the frozen number', 'tsb', 'middle')
    f.text(455, 106, 'what it costs you', 'tsb', 'middle')
    f.text(855, 106, 'where we remove it', 'tsb', 'middle')

    rows = [
        (125, 'C', 'channels it can read',
         'One model per sensor. RGB weights physically',
         'cannot load into a 32-band model.', 'PART 2', ORANGE, ORANGE_F),
        (255, '12', 'distinct blocks stored',
         '27.43 M parameters, every block stored',
         'separately — 98 % of the whole model.', 'PART 4', PURPLE, PURPLE_F),
    ]
    for y, num, cap, l1, l2, part, col, colf in rows:
        f.box(40, y, 160, 96, colf, col)
        f.text(120, y + 52, num, 'huge', 'middle', fill=col)
        f.text(120, y + 78, cap, 'ts', 'middle')

        f.line(208, y + 48, 258, y + 48, col)
        f.box(266, y, 380, 96, WHITE, GREY)
        f.text(456, y + 40, l1, 't', 'middle')
        f.text(456, y + 62, l2, 't', 'middle')

        f.line(654, y + 48, 704, y + 48, col)
        f.box(712, y, 288, 96, colf, col)
        f.text(856, y + 45, part, 'tb', 'middle', fill=col, size=19)
        f.text(856, y + 70, 'we delete it', 'ts', 'middle')

    f.banner(40, 372, 960,
             'PART 3 is the tool Part 4 needed — someone else’s paper, and none of it is ours.',
             PANEL, GREY, h=42)
    f.save()


# ================================================= 22 · three names, one lineage
def fig22():
    f = Fig('fig22_three_models.svg', 1040, 430)
    f.head('Three models, three names',
           'Two of them are ours. Each is the one before it with exactly one thing replaced.')

    specs = [
        (30, 'MedMamba', 'published baseline [1]', GREY, PANEL,
         ['4-stage hierarchy', '12 distinct blocks', 'RGB stem, C welded in'], '≈ 3.65 M'),
        (370, 'MedMamba-SS', 'our change ①', ORANGE, ORANGE_F,
         ['same hierarchy', '+ spectral pathway', 'C is gone'], '27.43 M'),
        (710, 'MedMamba-SS-TRM', 'our change ②', PURPLE, PURPLE_F,
         ['hierarchy → one core', '2 blocks, called 63×', 'C still gone'], '0.447 M'),
    ]
    for x, name, kind, col, colf, bullets, params in specs:
        f.box(x, 96, 300, 250, colf, col)
        f.text(x + 150, 128, name, 'tb', 'middle', size=17)
        f.text(x + 150, 149, kind, 'ts', 'middle', fill=col)
        yy = 184
        for b in bullets:
            f.circle(x + 40, yy - 5, 3, col, col)
            f.text(x + 54, yy, b, 't', 'start')
            yy += 28
        f.box(x + 40, 282, 220, 44, WHITE, col)
        f.text(x + 150, 302, params, 'tb', 'middle', size=17, fill=col)
        f.text(x + 150, 320, 'trainable parameters', 'ts', 'middle')

    f.line(334, 221, 362, 221, ORANGE)
    f.text(348, 205, 'add', 'ts', 'middle')
    f.line(674, 221, 702, 221, PURPLE)
    f.text(688, 205, 'swap', 'ts', 'middle')

    f.banner(30, 366, 980,
             'Older diagrams call the third one G-MedMamba-R. Same model, earlier name.',
             PANEL, GREY, h=40, cls='t')
    f.save()


# ================================================== 23 · the clinical task
def fig23():
    f = Fig('fig23_the_task.svg', 1040, 400)
    f.head('The job: sort breast tissue into three classes',
           'Two of them are easy. The whole difficulty — and the whole clinical stake — is in the third.')

    cards = [
        (40, 'healthy', 'no carcinoma', '24 % of test patches', GREEN, GREEN_F,
         'both models get this mostly right'),
        (370, 'DCIS', 'carcinoma, still contained', '7 % of test patches', RED, RED_F,
         'THE hard call — and the one that governs treatment'),
        (700, 'IDC', 'carcinoma, now invasive', '69 % of test patches', GREY, PANEL,
         'F1 = 0.999 for every model we tried'),
    ]
    for x, name, desc, share, col, colf, foot in cards:
        f.box(x, 92, 300, 170, colf, col)
        f.text(x + 150, 128, name, 'tb', 'middle', size=22, fill=col)
        f.text(x + 150, 156, desc, 't', 'middle')
        f.text(x + 150, 182, share, 'ts', 'middle')
        f.line(x + 60, 198, x + 240, 198, col, sw=1, arrow=False, dash='4 3')
        f.text(x + 150, 224, foot[:46], 'ts', 'middle')
        if len(foot) > 46:
            f.text(x + 150, 242, foot[46:], 'ts', 'middle')

    f.line(340, 177, 366, 177, GREY, sw=1.4)
    f.line(670, 177, 696, 177, GREY, sw=1.4)

    f.banner(40, 286, 960,
             'Because IDC is 69 % of the test set and everybody solves it, raw accuracy hides the result.',
             PANEL, GREY, h=42)
    f.banner(40, 340, 960,
             'Every number in this talk is BALANCED accuracy or macro-F1 — which do not.',
             GREEN_F, GREEN, h=42)
    f.save()


# ============================================= 24 · what a spectral camera adds
def fig24():
    f = Fig('fig24_hsi_vs_rgb.svg', 1040, 440)
    f.head('What the sensor actually hands you',
           'Same tissue, same position. One of them integrates the chemistry away.')

    # --- RGB panel
    f.box(30, 88, 470, 260, WHITE, GREY, dash='5 4', sw=1.2)
    f.text(265, 116, 'RGB camera — 3 numbers per pixel', 'tb', 'middle')
    for i, (c, lab) in enumerate([('#b4553f', 'R'), ('#4e8c6a', 'G'), ('#4a6ea8', 'B')]):
        f.box(70 + i * 76, 140, 60, 40, c, c, rx=5)
        f.text(100 + i * 76, 166, lab, 'tb', 'middle', fill=WHITE, size=17)
    # 3-point "spectrum"
    ax_x, ax_y, ax_w, ax_h = 70, 312, 390, 68
    f.line(ax_x, ax_y, ax_x + ax_w, ax_y, GREY, sw=1.2, arrow=False)
    for i, (px, py) in enumerate([(0.18, 0.55), (0.50, 0.75), (0.82, 0.40)]):
        cx = ax_x + px * ax_w
        f.line(cx, ax_y, cx, ax_y - py * ax_h, GREY, sw=8, arrow=False)
        f.circle(cx, ax_y - py * ax_h, 5, GREY, GREY)
    f.text(265, 204, 'three wide, overlapping buckets', 'ts', 'middle')
    f.text(265, 224, 'anything narrower than a bucket is averaged out', 'ts', 'middle')
    f.text(265, 324, '450 nm  →  700 nm', 'ts', 'middle')

    # --- HSI panel
    f.box(540, 88, 470, 260, ORANGE_F, ORANGE, dash='5 4', sw=1.2)
    f.text(775, 116, 'Hyperspectral cube — 32 numbers per pixel', 'tb', 'middle')
    for i in range(32):
        f.box(566 + i * 13.4, 140, 10, 40, WHITE if i % 4 else ORANGE, ORANGE, rx=2, sw=0.9)
    # a real-looking spectrum
    pts = [0.30, 0.34, 0.41, 0.52, 0.63, 0.70, 0.66, 0.55, 0.47, 0.44, 0.48, 0.58,
           0.70, 0.80, 0.86, 0.88, 0.84, 0.78, 0.74, 0.73, 0.76, 0.81, 0.86, 0.90,
           0.92, 0.91, 0.87, 0.80, 0.71, 0.62, 0.54, 0.49]
    bx, by, bw, bh = 580, 312, 390, 68
    f.line(bx, by, bx + bw, by, GREY, sw=1.2, arrow=False)
    d = ' '.join(f'{"M" if i == 0 else "L"}{bx + i * bw / 31:.1f},{by - v * bh:.1f}'
                 for i, v in enumerate(pts))
    f.path(d, ORANGE, sw=2.2, arrow=False)
    for i, v in enumerate(pts):
        f.circle(bx + i * bw / 31, by - v * bh, 2.6, ORANGE, ORANGE, sw=0)
    f.text(775, 204, 'a real spectrum, sampled 32 times', 'ts', 'middle')
    f.text(775, 224, 'absorption depends on molecular composition', 'ts', 'middle')
    f.text(775, 324, '450 nm  →  950 nm   (32 bands chosen from 740)', 'ts', 'middle')

    f.banner(30, 380, 980,
             'Whether that extra chemistry helps a diagnosis is an empirical question. Part 6 is our answer, '
             'for one dataset.', PANEL, GREY, h=42, cls='t')
    f.save()


# ================================================== 25 · the parameter proof
def fig25():
    f = Fig('fig25_param_proof.svg', 1040, 400)
    f.head('The proof: the same model, at 32 bands and at 3',
           'Same dataset, same three classes, same everything. Only the input directory differs.')

    for x, bands, cap in [(70, '32 bands', 'hyperspectral cube'),
                          (610, '3 bands', 'RGB rendering')]:
        f.box(x, 100, 360, 170, GREEN_F, GREEN)
        f.text(x + 180, 136, bands, 'tb', 'middle', size=19)
        f.text(x + 180, 158, cap, 'ts', 'middle')
        f.text(x + 180, 212, '446,409', 'huge', 'middle', fill=GREEN)
        f.text(x + 180, 240, 'trainable parameters', 'ts', 'middle')

    f.text(520, 200, '=', 'huge', 'middle', fill=GREY, size=52)
    f.text(520, 236, 'difference: 0', 'tsb', 'middle')

    f.banner(70, 292, 900,
             'Not “about the same”. Identical — this is arithmetic, not a measurement.',
             GREEN_F, GREEN, h=44)
    f.text(520, 368, 'The honest footnote: STORAGE is constant in band count. ACTIVATIONS are not '
                     '— peak inference memory is 170 MB vs 63 MB.', 'ts', 'middle')
    f.save()


# =============================================== 26 · where the parameters live
def fig26():
    f = Fig('fig26_where_params_live.svg', 1040, 450)
    f.head('C is gone. And it barely dented the model.',
           'Everything Part 2 spent five minutes on is the thin sliver at the top.')

    f.text(700, 110, 'share of MedMamba-SS', 'tsb', 'middle')
    rows = [
        ('the spectral front end — all of Part 2', 38724 / 27430000, ORANGE, ORANGE_F, '38,724', '0.1 %'),
        ('everything else \u2014 backbone, fusion, heads', 1.0, GREY, BAR_GREY, '27,391,276', '99.9 %'),
    ]
    y = 140
    for lab, frac, col, colf, val, pct in rows:
        f.text(30, y + 26, lab, 't', 'start')
        f.box(30, y + 40, 700, 42, PANEL, '#e3e6ea', rx=5, sw=1)
        f.box(30, y + 40, max(4.0, 700 * frac), 42, colf, col, rx=5, sw=1.6)
        f.text(748, y + 60, val, 'tb', 'start', size=17)
        f.text(748, y + 80, pct, 'ts', 'start')
        y += 106

    f.banner(30, 348, 980,
             'So: can a backbone REUSE the same weights instead of storing twelve sets?',
             PURPLE_F, PURPLE, h=46, cls='tb', color=INK)
    f.text(520, 424, 'That has an answer in a paper from October 2025 — and a bigger one '
                     'than we went looking for.', 'ts', 'middle')
    f.save()


# ============================================ 27 · y is the answer, z the working
def fig27():
    f = Fig('fig27_y_and_z.svg', 1040, 460)
    f.head('Two things are carried between passes, and only one is readable',
           'This is the whole state of the model. There is nothing else.')

    def grid(x, y, cells, col):
        for r in range(3):
            for c in range(3):
                v = cells[r * 3 + c]
                f.box(x + c * 30, y + r * 30, 28, 28, WHITE if v else PANEL, col, rx=3, sw=1.1)
                if v:
                    f.text(x + c * 30 + 14, y + r * 30 + 20, v, 'monb', 'middle')

    # --- y
    f.box(30, 92, 470, 268, GREEN_F, GREEN)
    f.text(265, 124, 'y  —  the answer', 'tb', 'middle', size=17, fill=GREEN)
    f.text(265, 148, 'push it through the output head, any time you like', 'ts', 'middle')
    grid(120, 168, ['5', '3', '', '6', '', '2', '', '9', '8'], GREY)
    f.line(228, 213, 268, 213, GREEN)
    grid(288, 168, ['5', '3', '4', '6', '7', '2', '1', '9', '8'], GREEN)
    f.text(174, 285, 'mid-recursion', 'ts', 'middle')
    f.text(342, 285, 'a few passes later', 'ts', 'middle')
    f.banner(60, 302, 410, 'A real, gradable Sudoku grid — getting better.',
             WHITE, GREEN, h=40, cls='t')

    # --- z
    f.box(540, 92, 470, 268, PANEL, GREY, dash='5 4', sw=1.2)
    f.text(775, 124, 'z  —  the working', 'tb', 'middle', size=17, fill=GREY)
    f.text(775, 148, 'push it through the same head and you get…', 'ts', 'middle')
    import random
    random.seed(11)
    for r in range(3):
        for c in range(3):
            f.box(630 + c * 30, 168 + r * 30, 28, 28, WHITE, GREY, rx=3, sw=1.1)
            f.text(644 + c * 30, 188 + r * 30, random.choice('?~··?~'), 'mono', 'middle')
    f.line(738, 213, 778, 213, GREY)
    f.text(870, 205, 'nothing meaningful.', 't', 'middle')
    f.text(870, 227, 'It is not a solution.', 't', 'middle')
    f.text(775, 285, 'nobody — not even the loss — ever looks at it', 'ts', 'middle')
    f.banner(570, 302, 410, 'Scratch paper. And it is why the model can improve.',
             WHITE, GREY, h=40, cls='t')

    # --- the ablation, as bars
    f.text(30, 396, 'Why exactly two? The paper removed each one — Sudoku-Extreme accuracy:',
           'tsb', 'start')
    bars = [('carry y and z  (as published)', 87.4, GREEN, GREEN_F),
            ('split z into 7 features', 77.6, GREY, BAR_GREY),
            ('carry z only — no answer to build on', 71.9, RED, RED_F)]
    for i, (lab, v, col, colf) in enumerate(bars):
        yy = 412 + i * 16
        f.text(340, yy + 4, lab, 'ts', 'end')
        f.box(350, yy - 6, 460, 12, PANEL, '#e3e6ea', rx=3, sw=0.8)
        f.box(350, yy - 6, 460 * v / 100, 12, colf, col, rx=3, sw=1.2)
        f.text(820, yy + 4, f'{v} %', 'tsb', 'start')
    f.save()


# ================================================== 28 · the recursion, precisely
def fig28():
    f = Fig('fig28_code.svg', 1040, 470)
    f.head('The same idea, precisely — and it is four lines',
           'Everything TRM does that matters is on this slide.')

    f.box(30, 88, 620, 150, '#f6f5fa', '#ddd8e8', rx=8, sw=1.2)
    code = [
        ('for i in range(6):', 0, 0),
        ('z = f(z + y + x)', 1, 22),      # thinking - indented by x, not by spaces
        ('y = f(y + z)', 2, 0),           # answering
    ]
    yy = 120
    for line, kind, indent in code:
        col = ORANGE if kind == 1 else (GREEN if kind == 2 else INK)
        f.text(56 + indent, yy, line, 'monb' if kind else 'mono', 'start', fill=col, size=17)
        if kind == 1:
            f.text(330, yy, '←  six scribbles: update the working', 'ts', 'start')
        if kind == 2:
            f.text(330, yy, '←  one stroke: update the answer', 'ts', 'start')
        yy += 40
    f.text(56, 222, 'that is ONE round.  Three rounds is one pass.', 'ts', 'start')

    # the three things to notice
    items = [
        (1, GREY, 'Both calls are the SAME network.',
         'Same weights, called again. That is the entire parameter saving.'),
        (2, GREY, 'Inputs are ADDED, never concatenated.',
         'So f’s input is the same shape on call 1 and on call 63.'),
        (3, ORANGE, 'x is in the thinking line and MISSING from the answering line.',
         'That is the only thing telling one shared network which job to do.'),
    ]
    yy = 266
    for n, col, bold, sub in items:
        f.badge(48, yy + 2, n, col)
        f.text(72, yy + 7, bold, 'tb', 'start')
        f.text(72, yy + 28, sub, 'ts', 'start')
        yy += 56

    # the measured payoff
    f.box(676, 88, 334, 150, PANEL, GREY)
    f.text(843, 116, 'one network or two?', 'tsb', 'middle')
    for i, (lab, acc, prm, col) in enumerate(
            [('two separate networks', '82.4 %', '10 M', GREY),
             ('one shared f', '87.4 %', '5 M', GREEN)]):
        y0 = 132 + i * 50
        f.box(694, y0, 298, 42, WHITE if i == 0 else GREEN_F, col, rx=5, sw=1.4)
        f.text(708, y0 + 26, lab, 't', 'start')
        f.text(916, y0 + 26, acc, 'tb', 'middle', fill=col)
        f.text(972, y0 + 26, prm, 'ts', 'middle')
    f.text(843, 256, 'Half the parameters, and five points better.', 'tb', 'middle')

    f.banner(676, 276, 334,
             'See x → you are thinking.', ORANGE_F, ORANGE, h=42, cls='tb')
    f.banner(676, 326, 334,
             'No x → you are answering.', GREEN_F, GREEN, h=42, cls='tb')
    f.text(843, 404, 'Point at the two lines. This is the one people miss.', 'ts', 'middle')
    f.save()


# ================================================== 29 · does TRM actually work
def fig29():
    f = Fig('fig29_trm_benchmarks.svg', 1040, 440)
    f.head('Does it work? Yes — and here is the row that keeps you honest',
           '7 M parameters, about 1000 training examples. Every number is the paper’s.')

    models = [('Deepseek R1', '671 B', GREY, BAR_GREY),
              ('HRM', '27 M', GREY, BAR_GREY),
              ('TRM', '7 M', PURPLE, PURPLE_F),
              ('Grok-4-thinking', '1.7 T', RED, RED_F)]
    tasks = [('Sudoku-Extreme', [0.0, 55.0, 87.4, None]),
             ('Maze-Hard', [0.0, 74.5, 85.3, None]),
             ('ARC-AGI-1', [15.8, 40.3, 44.6, 66.7])]

    x0, gw = 92, 300
    for ti, (task, vals) in enumerate(tasks):
        gx = x0 + ti * gw
        f.text(gx + 110, 108, task, 'tb', 'middle')
        for mi, v in enumerate(vals):
            name, size, col, colf = models[mi]
            yy = 128 + mi * 44
            f.text(gx + 92, yy + 22, f'{name}  \u00b7  {size}' if ti == 0 else name,
                   'ts', 'end')
            if v is None:
                f.text(gx + 104, yy + 22, 'not reported', 'ts', 'start')
                continue
            f.box(gx + 100, yy + 8, 120, 20, PANEL, '#e3e6ea', rx=3, sw=0.8)
            f.box(gx + 100, yy + 8, max(2.0, 120 * v / 100), 20, colf, col, rx=3, sw=1.3)
            f.text(gx + 228, yy + 23, f'{v}', 'tsb', 'start')

    f.banner(30, 316, 486,
             'On Sudoku and Maze every LLM here scores a flat 0.0.',
             GREEN_F, GREEN, h=44, cls='t')
    f.banner(524, 316, 486,
             'Grok-4 beats it on ARC. The abstract says “most LLMs” and means it.',
             RED_F, RED, h=44, cls='t')

    f.text(520, 392, 'Not generative — one input, one deterministic answer.   '
                     'And bigger is worse: 4 layers lose to 2, and the authors say they have no theory why.',
           'ts', 'middle')
    f.text(520, 416, 'None of this is measured on medical images. We inherited the recursion, not the evidence.',
           'tsb', 'middle')
    f.save()


# ============================================ 30 · what to carry out of Part 3
def fig30():
    f = Fig('fig30_trm_carry.svg', 1040, 370)
    f.head('Three things to carry into Part 4',
           'If you remember nothing else from the last twelve minutes, remember these.')

    cards = [
        (1, 'Refinement beats capacity',
         ['The model is never asked “solve this”.', 'It is asked “here is a wrong answer —',
          'make it less wrong”. So it never needs', 'the capacity to be right in one shot.'], PURPLE, PURPLE_F),
        (2, 'One network, two jobs',
         ['z = f(z+y+x) thinks.', 'y = f(y+z) answers.', 'Whether it can see x is the only',
          'thing that tells it which job it is on.'], ORANGE, ORANGE_F),
        (3, 'Differentiate a whole round',
         ['Run 3 rounds, the first 2 without', 'gradients. Backpropagate one complete',
          'round, not one step. Worth 31 points —', 'the largest single effect in the paper.'], GREEN, GREEN_F),
    ]
    for i, (n, head, lines, col, colf) in enumerate(cards):
        x = 30 + i * 333
        f.box(x, 92, 314, 212, colf, col)
        f.badge(x + 28, 122, n, col)
        f.text(x + 50, 128, head, 'tb', 'start', size=15)
        yy = 168
        for ln in lines:
            f.text(x + 22, yy, ln, 'ts', 'start')
            yy += 22

    f.banner(30, 320, 980,
             'Everything from here is ours again.', PANEL, GREY, h=40, cls='tb')
    f.save()


# ================================================== 31 · what we changed vs TRM
def fig31():
    f = Fig('fig31_what_we_changed.svg', 1040, 470)
    f.head('What we changed from TRM, and why',
           'Five changes. We kept every part that Part 3 showed was load-bearing.')

    f.text(232, 106, 'TRM', 'tsb', 'middle')
    f.text(560, 106, 'ours', 'tsb', 'middle')
    f.text(840, 106, 'why', 'tsb', 'middle')

    rows = [
        ('shape', 'token sequence', '2-D grid', 'we classify images'),
        ('mixer inside f', 'self-attention', 'depthwise 3×3 conv',
         'attention is overkill on 11×11 — and'),
        ('passes', '16', '3', '63 calls to f, not 336'),
        ('deep supervision', '1 pass per step', 'all passes in one forward',
         'keeps our per-batch checks working'),
        ('halting', 'per sample, on', 'whole batch, OFF',
         'sat at chance for 33 epochs'),
    ]
    y = 122
    for i, (knob, trm, ours, why) in enumerate(rows):
        f.text(30, y + 26, knob, 'tb', 'start', size=14)
        f.box(146, y + 6, 172, 34, PANEL, GREY, rx=5, sw=1.2)
        f.text(232, y + 28, trm, 'ts', 'middle')
        f.line(326, y + 23, 352, y + 23, ORANGE, sw=1.5)
        f.box(360, y + 6, 200, 34, ORANGE_F, ORANGE, rx=5, sw=1.4)
        f.text(460, y + 28, ours, 'ts', 'middle')
        f.text(586, y + 28, why, 'ts', 'start')
        if i == 1:
            f.text(586, y + 44, 'their own ablation says the mixer is task-dependent', 'ts', 'start')
        y += 54

    f.banner(30, 396, 980,
             'UNCHANGED: both lines, one shared network, additive inputs, 2 rounds under no_grad with a '
             'FULL round backpropagated, detach between passes.',
             GREEN_F, GREEN, h=40, cls='t')
    f.text(30, 456, 'Say it before anyone catches you: we implemented TRM’s RECURSION, not TRM’s BLOCKS. '
                    'They use attention; every real run of ours uses a convolution.', 'tsb', 'start')
    f.save()


# ================================================== 32 · the model, simply
def fig32():
    f = Fig('fig32_pipeline_simple.svg', 1040, 380)
    f.head('MedMamba-SS-TRM, end to end',
           'Five boxes. Notice what is not here: no stages, no patch merging, no downsampling.')

    boxes = [
        (36, 120, 'H × W × C', 'any band count', WHITE, GREY),
        (214, 174, 'Spectral Pathway', 'C disappears here', ORANGE_F, ORANGE),
        (420, 130, 'Stem', 'Linear(64→98)', WHITE, GREY),
        (582, 186, 'Recursive Core', '2 layers, called 63×', PURPLE_F, PURPLE),
        (800, 130, 'Head', 'norm → pool → linear', WHITE, GREY),
    ]
    for x, w, name, sub, fill, col in boxes:
        f.box(x, 116, w, 96, fill, col)
        f.text(x + w / 2, 158, name, 'tb', 'middle')
        f.text(x + w / 2, 180, sub, 'ts', 'middle')
    for x1, x2 in [(156, 206), (392, 412), (554, 574), (772, 792)]:
        f.line(x1, 164, x2, 164, GREY)
    f.line(946, 164, 990, 164, GREY)
    f.text(1004, 168, '3', 'tb', 'middle', size=17)
    f.text(1004, 190, 'classes', 'ts', 'middle')

    f.text(262, 236, 'mean over the BAND axis', 'ts', 'start', fill=ORANGE)
    f.line(300, 244, 300, 218, ORANGE, sw=1.4)
    f.text(608, 236, 'width 128 from the stem to the head', 'ts', 'start', fill=PURPLE)
    f.line(672, 244, 672, 218, PURPLE, sw=1.4)

    f.banner(36, 264, 468,
             'Same front end, same head, same trainer as the hierarchical version.',
             PANEL, GREY, h=42, cls='t')
    f.banner(524, 264, 486,
             'Every call to f is gradient-checkpointed — the only reason it fits on the card.',
             RED_F, RED, h=42, cls='t')
    f.text(520, 344, 'One config flag switches between the 12-block hierarchy and this. '
                     'Everything else is byte-identical.', 'ts', 'middle')
    f.save()


# ============================================== 33 · against a MedMamba baseline
def fig33():
    f = Fig('fig33_vs_medmamba.svg', 1040, 512)
    f.head('Against a MedMamba baseline 8× its size',
           'Identical test patches, identical patient split, identical seed, batch size and GPU.')

    metrics = [
        ('trainable parameters', 0.446, 3.65, '0.446 M', '3.65 M', True),
        ('balanced accuracy', 90.73, 86.76, '90.73 %', '86.76 %', False),
        ('macro-F1', 0.8580, 0.8278, '0.858', '0.828', False),
        ('F1 on DCIS — the hard class', 0.704, 0.637, '0.704', '0.637', False),
        ('seconds / epoch', 679, 155, '679 s', '155 s', True),
    ]
    # one legend instead of repeating the two model names on every row
    for i, (name, col, colf) in enumerate([('MedMamba-SS-TRM  (0.45 M)', PURPLE, PURPLE_F),
                                           ('MedMamba  (3.65 M)', GREY, PANEL)]):
        lx = 636 + i * 200
        f.box(lx, 74, 18, 12, colf, col, rx=3, sw=1.3)
        f.text(lx + 26, 85, name, 'ts', 'start')

    y = 104
    for lab, ours, base, o_s, b_s, lower_better in metrics:
        f.text(30, y + 20, lab, 't', 'start')
        top = max(ours, base)
        for i, (v, s_, col, colf) in enumerate([(ours, o_s, PURPLE, PURPLE_F),
                                                (base, b_s, GREY, BAR_GREY)]):
            yy = y + i * 24
            good = (v < top) if lower_better else (v >= top)
            f.box(400, yy, 400, 18, PANEL, '#e3e6ea', rx=3, sw=0.8)
            f.box(400, yy, max(3.0, 400 * v / top), 18, colf, col, rx=3, sw=1.3)
            f.text(810, yy + 14, s_, 'tsb' if good else 'ts', 'start')
        if lower_better:
            f.text(950, y + 20, 'lower is better', 'ts', 'start', fill=MUTED, size=11)
        y += 62

    f.banner(30, 412, 490,
             'Leads every aggregate metric, at an eighth of the parameters.',
             GREEN_F, GREEN, h=42, cls='t')
    f.banner(540, 412, 470,
             'And costs 4.4× the wall-clock to get there.',
             RED_F, RED, h=42, cls='t')
    f.text(520, 486, 'Read this as INDICATIVE, not controlled: the two models trained on differently '
                     'balanced splits, and it is one seed.', 'tsb', 'middle')
    f.save()


# ================================================== 34 · what we have not measured
def fig34():
    f = Fig('fig34_not_measured.svg', 1040, 420)
    f.head('Three questions you are entitled to ask',
           'Volunteering these costs a minute and turns the three hardest questions into things we said first.')

    qs = [
        ('Does the recursion help\nYOUR model?',
         'We don’t know.',
         ['No depth ablation. No hierarchical-versus-recursive',
          'run under matched conditions. It is one flag away',
          'and has never been run.'], RED),
        ('Do you inherit\nTRM’s results?',
         'No.',
         ['Theirs are puzzle grids with a verifiable answer and',
          '1000× augmentation. We inherited the recursion',
          'SCHEDULE, not the evidence.'], RED),
        ('Why a convolution\nand not SS2D?',
         'Speed.',
         ['Our selective scan is a pure-PyTorch loop — about',
          '39× slower at dataset scale. ss2d is implemented',
          'and has never been used in a real run.'], ORANGE),
    ]
    y = 96
    for q1, ans, lines, col in qs:
        a, b = q1.split('\n')
        f.box(30, y, 250, 86, PANEL, GREY)
        f.text(155, y + 36, a, 'tb', 'middle', size=14)
        f.text(155, y + 58, b, 'tb', 'middle', size=14)
        f.line(288, y + 43, 314, y + 43, col)
        f.box(322, y, 688, 86, WHITE, col)
        f.text(342, y + 28, ans, 'tb', 'start', size=17, fill=col)
        yy = y + 24
        for ln in lines:
            f.text(516, yy, ln, 'ts', 'start')
            yy += 20
        y += 100

    f.banner(30, 388, 980,
             'Two standing weaknesses: every result is a SINGLE SEED, and five test patients.',
             RED_F, RED, h=32, cls='t')
    f.save()


# =================================================== 35 · the five takeaways
def fig35():
    f = Fig('fig35_takeaways.svg', 1040, 486)
    f.head('Five things to take away', 'Number three is the one we would keep.')

    items = [
        (1, 'Delete C from your weight shapes.', ORANGE,
         'A shared per-value projection plus a physical wavelength tag. One model then serves every sensor.'),
        (2, 'TRM is two lines.', PURPLE,
         'z = f(z+y+x) six times, y = f(y+z) once, most of it under no_grad.'),
        (3, 'Refinement beats capacity.', GREEN,
         'A model that only has to make an answer LESS WRONG can be far smaller than one that must be '
         'right in one shot. It borrows what it is missing from time.'),
        (4, 'Weight sharing trades storage for compute — steeply.', RED,
         '8× smaller, 191× more arithmetic, and gradient checkpointing became mandatory. '
         'Always report both numbers.'),
        (5, 'Be precise about provenance.', GREY,
         'We implemented TRM’s RECURSION, not TRM’s BLOCKS.'),
    ]
    y = 92
    for n, head, col, sub in items:
        h = 76 if n in (3, 4) else 62
        f.box(30, y, 980, h, WHITE if n != 3 else GREEN_F, col)
        f.badge(62, y + 30, n, col)
        f.text(90, y + 28, head, 'tb', 'start', size=16)
        if len(sub) > 104:
            cut = sub.rfind(' ', 0, 104)
            f.text(90, y + 50, sub[:cut], 'ts', 'start')
            f.text(90, y + 68, sub[cut + 1:], 'ts', 'start')
        else:
            f.text(90, y + 50, sub, 'ts', 'start')
        y += h + 8
    f.save()


# ======================================= 36 · where TRM lands in our model
def fig36():
    f = Fig('fig36_integration.svg', 1040, 450)
    f.head('Where TRM actually lands',
           'One of the three parts is replaced. The other two are reused byte for byte.')

    rows = [
        ('Spectral Pathway', 'band-agnostic front end', ORANGE, ORANGE_F,
         'Spectral Pathway', 'band-agnostic front end', ORANGE, ORANGE_F, 'kept', GREEN),
        ('4-stage hierarchy', '12 distinct blocks · 27 M', GREY, PANEL,
         'Recursive Core', '2 blocks, called 63× · 0.40 M', PURPLE, PURPLE_F, 'SWAPPED', RED),
        ('Head', 'norm → pool → linear', GREY, PANEL,
         'Head', 'norm → pool → linear', GREY, PANEL, 'kept', GREEN),
    ]
    f.text(190, 106, 'MedMamba-SS', 'tb', 'middle', size=16)
    f.text(790, 106, 'MedMamba-SS-TRM', 'tb', 'middle', size=16)

    y = 124
    for ln, ls, lc, lf, rn, rs, rc, rf, verdict, vc in rows:
        f.box(40, y, 300, 72, lf, lc)
        f.text(190, y + 32, ln, 'tb', 'middle')
        f.text(190, y + 54, ls, 'ts', 'middle')
        f.box(640, y, 300, 72, rf, rc)
        f.text(790, y + 32, rn, 'tb', 'middle')
        f.text(790, y + 54, rs, 'ts', 'middle')
        f.line(352, y + 36, 628, y + 36, vc if verdict == 'SWAPPED' else GREY,
               sw=2.2 if verdict == 'SWAPPED' else 1.4,
               dash=None if verdict == 'SWAPPED' else '5 4')
        f.text(490, y + 26, verdict, 'tb' if verdict == 'SWAPPED' else 'ts', 'middle', fill=vc)
        y += 88

    f.text(490, 106, 'one config flag', 'tsb', 'middle')
    f.banner(40, 392, 900,
             'The swap is legal only because the interface did not move: same input shape, same '
             'output keys, same head, same trainer.',
             PANEL, GREY, h=44, cls='t')
    f.save()


# ==================================== 37 · TRM’s four symbols, in our tensors
def fig37():
    f = Fig('fig37_trm_mapping.svg', 1040, 500)
    f.head('TRM’s x, y, z and f — in our model',
           'The recursion is theirs. These four lines are what it is actually operating on.')

    f.text(66, 106, 'TRM', 'tsb', 'start')
    f.text(250, 106, 'in our model', 'tsb', 'start')
    f.text(848, 106, 'shape', 'tsb', 'middle')

    rows = [
        ('x', 'Stem(SpectralPathway(patch))', 'computed ONCE, then held fixed for all 63 calls',
         '[Hp, Wp, 128]', ORANGE, ORANGE_F),
        ('y', 'the feature grid the head reads', 'decode it at any segment and you get logits',
         '[Hp, Wp, 128]', GREEN, GREEN_F),
        ('z', 'the latent working', 'same shape, never decoded, never supervised',
         '[Hp, Wp, 128]', GREY, PANEL),
        ('f', '2 × (mixer → GatedMLP)', 'post-norm RMSNorm · the only weights in the core',
         '398,592 params', PURPLE, PURPLE_F),
    ]
    y = 120
    for sym, what, note, shape, col, colf in rows:
        f.box(40, y, 56, 60, colf, col)
        f.text(68, y + 40, sym, 'monb', 'middle', fill=col, size=26)
        f.text(116, y + 26, what, 'tb', 'start', size=14)
        f.text(116, y + 48, note, 'ts', 'start')
        f.box(760, y + 14, 176, 32, WHITE, col, rx=5, sw=1.3)
        f.text(848, y + 35, shape, 'mono', 'middle')
        y += 72

    f.banner(40, 420, 460,
             'y_init and z_init are BUFFERS, not parameters.', RED_F, RED, h=42, cls='t')
    f.banner(520, 420, 480,
             '(6 + 1) × 3 rounds × 3 passes = 63 calls to f.', PURPLE_F, PURPLE, h=42, cls='t')
    f.text(520, 484, 'The first read of the buffers is inside no_grad, so a parameter there '
                     'would silently never receive a gradient.', 'ts', 'middle')
    f.save()


# ================================= 38 · MedMamba, saying what actually happens
def fig38():
    f = Fig('fig38_medmamba_clear.svg', 1040, 580)
    f.head('MedMamba — what each step actually does',
           'Two operations, repeated. One shrinks the grid; the other is the block that does the work.')

    # ---------- the pipeline, with the grid drawn to scale
    f.text(30, 96, 'THE SHAPE', 'tsb', 'start')
    f.box(30, 108, 132, 92, RED_F, RED)
    f.text(96, 132, 'cut into', 'ts', 'middle')
    f.text(96, 152, '4×4 patches', 'tb', 'middle', size=13.5)
    f.text(96, 173, 'each → 96 numbers', 'ts', 'middle')
    f.text(96, 193, 'the 3 lives here', 'tsb', 'middle', fill=RED)

    stages = [(186, 56, 96, 2, 44), (386, 28, 192, 2, 34), (586, 14, 384, 4, 26), (786, 7, 768, 2, 20)]
    for i, (x, g, ch, nb, px) in enumerate(stages):
        f.box(x, 108, 150, 92, PANEL, GREY)
        f.text(x + 75, 130, f'stage {i + 1}', 'tb', 'middle', size=13.5)
        # the grid, drawn to scale
        f.box(x + 12, 140, px, px, WHITE, GREY, rx=2, sw=1.2)
        n = max(2, min(7, g // 8))
        for k in range(1, n):
            f.line(x + 12 + px * k / n, 140, x + 12 + px * k / n, 140 + px, '#c9ced6', sw=0.6, arrow=False)
            f.line(x + 12, 140 + px * k / n, x + 12 + px, 140 + px * k / n, '#c9ced6', sw=0.6, arrow=False)
        f.text(x + 22 + px, 156, f'{g}×{g} grid', 'ts', 'start')
        f.text(x + 22 + px, 174, f'{ch} numbers', 'ts', 'start')
        f.text(x + 75, 192, f'× {nb} blocks', 'tsb', 'middle')
        if i < 3:
            f.line(x + 154, 154, x + 182, 154, GREY)

    f.line(166, 154, 182, 154, GREY)
    f.box(30, 216, 906, 40, ORANGE_F, ORANGE)
    f.text(483, 241, 'BETWEEN STAGES — patch merging: glue each 2×2 group of cells into one, then '
                     'halve the result. Grid ÷ 2, numbers × 2.', 't', 'middle')
    f.box(952, 108, 58, 92, GREEN_F, GREEN)
    f.text(981, 146, 'average', 'ts', 'middle')
    f.text(981, 164, '+ one', 'ts', 'middle')
    f.text(981, 182, 'linear', 'ts', 'middle')
    f.line(940, 154, 948, 154, GREY, arrow=False)

    # ---------- what one block does
    f.text(30, 292, 'INSIDE ONE BLOCK — this is the part that does the work', 'tsb', 'start')
    f.box(30, 306, 118, 78, WHITE, GREY)
    f.text(89, 338, 'one cell,', 'ts', 'middle')
    f.text(89, 356, '96 numbers', 'tb', 'middle', size=13.5)
    f.line(152, 345, 184, 345, GREY)
    f.text(168, 330, 'split', 'ts', 'middle')

    f.box(192, 300, 300, 40, GREEN_F, GREEN)
    f.text(342, 325, 'first 48  →  SS2D', 'tb', 'middle', size=13.5)
    f.text(192, 360, 'reads the WHOLE grid, four directions,', 'ts', 'start')
    f.text(192, 378, 'and remembers what it passed — long range.', 'ts', 'start')

    f.box(192, 396, 300, 40, PANEL, GREY)
    f.text(342, 421, 'other 48  →  3×3 convolution', 'tb', 'middle', size=13.5)
    f.text(192, 456, 'sees only each cell’s neighbours — local detail.', 'ts', 'start')

    f.line(496, 320, 528, 348, GREY)
    f.line(496, 416, 528, 388, GREY)
    f.box(536, 336, 132, 64, WHITE, GREY)
    f.text(602, 362, 'glue back', 'ts', 'middle')
    f.text(602, 382, 'to 96', 'ts', 'middle')
    f.line(672, 368, 704, 368, GREY)
    f.box(712, 336, 150, 64, ORANGE_F, ORANGE)
    f.text(787, 362, 'shuffle the', 'ts', 'middle')
    f.text(787, 382, 'channels', 'tb', 'middle', size=13.5)
    f.line(866, 368, 898, 368, GREY)
    f.box(906, 336, 104, 64, WHITE, GREY)
    f.text(958, 372, 'out', 'tb', 'middle')

    f.text(712, 424, 'so the two halves swap jobs', 'ts', 'start')
    f.text(712, 442, 'on the next block', 'ts', 'start')

    f.banner(30, 486, 980,
             'A long-range operator and a local one for about the price of one. That economy is '
             'what we inherited — and did not touch.', PANEL, GREY, h=44, cls='t')
    f.text(520, 556, 'Everything on this slide is MedMamba’s. The only part we change is the '
                     'red box on the left.', 'tsb', 'middle')
    f.save()


# =========================================== 39 · HRM, the paper TRM starts from
def fig39():
    f = Fig('fig39_hrm.svg', 1040, 584)
    f.head('HRM — the paper TRM starts from',
           'Wang et al., 2025. You need one minute of it, because TRM is defined by what it deleted.')

    # --- what HRM is
    f.box(30, 92, 470, 168, PANEL, GREY)
    f.text(265, 118, 'What HRM does', 'tb', 'middle')
    f.box(66, 136, 170, 52, WHITE, PURPLE)
    f.text(151, 158, 'slow network', 'tb', 'middle', size=13.5)
    f.text(151, 178, 'updates rarely', 'ts', 'middle')
    f.box(66, 200, 170, 52, WHITE, PURPLE)
    f.text(151, 222, 'fast network', 'tb', 'middle', size=13.5)
    f.text(151, 242, 'updates often', 'ts', 'middle')
    f.path('M 240,162 C 280,162 280,226 244,226', PURPLE, sw=1.8)
    f.path('M 244,214 C 284,214 284,150 244,150', PURPLE, sw=1.8)
    f.text(330, 168, 'Two recurrent modules', 'ts', 'start')
    f.text(330, 186, 'at two timescales, each', 'ts', 'start')
    f.text(330, 204, '4 layers. 27 M parameters.', 'ts', 'start')
    f.text(330, 230, 'It beat the LLMs on Sudoku,', 'tsb', 'start')
    f.text(330, 248, 'mazes and ARC.', 'tsb', 'start')

    # --- what made engineers uneasy
    f.box(540, 92, 470, 168, RED_F, RED)
    f.text(775, 118, 'What it justified the design with', 'tb', 'middle')
    for i, (head, sub) in enumerate([
            ('Biology', 'arguments about timescales in the brain'),
            ('A fixed-point theorem', 'assume the loop settles, then backpropagate 2 steps of 6')]):
        y = 140 + i * 58
        f.box(566, y, 418, 48, WHITE, RED)
        f.text(584, y + 21, head, 'tb', 'start', size=13.5)
        f.text(584, y + 39, sub, 'ts', 'start')
    f.text(775, 282, 'TRM’s objection: the loop never actually settles — it just stops.', 'tsb', 'middle')

    # --- the deletions
    f.text(30, 320, 'TRM IS HRM WITH THINGS DELETED — and every deletion made it better',
           'tsb', 'start')
    rows = [('two networks at two timescales', 'one network'),
            ('4 layers each', '2 layers'),
            ('biology and a fixed-point theorem', 'y = the answer, z = the working'),
            ('27 M parameters', '7 M — and every benchmark goes UP')]
    y = 334
    for a, b in rows:
        f.box(30, y, 390, 32, PANEL, GREY, rx=5, sw=1.2)
        f.text(225, y + 21, a, 'ts', 'middle')
        f.line(428, y + 16, 456, y + 16, PURPLE, sw=1.5)
        f.box(464, y, 390, 32, PURPLE_F, PURPLE, rx=5, sw=1.2)
        f.text(659, y + 21, b, 'ts', 'middle')
        y += 38

    f.banner(30, 494, 980,
             'An independent audit found HRM’s recursion was worth almost nothing: 35.7 % → 39.0 %.',
             RED_F, RED, h=40, cls='t')
    f.text(520, 562, 'TRM’s claim is not that the recursion was a weak idea. It is that it was '
                     'implemented wrong.', 'tsb', 'middle')
    f.save()


if __name__ == '__main__':
    print('writing v2 figures:')
    for fn in (fig21, fig22, fig23, fig24, fig25, fig26, fig27, fig28,
               fig29, fig30, fig31, fig32, fig33, fig34, fig35, fig36, fig37, fig38, fig39):
        fn()
    print('all bounds checks passed')
