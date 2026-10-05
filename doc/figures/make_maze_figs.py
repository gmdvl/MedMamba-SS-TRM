#!/usr/bin/env python3
"""The two TRM maze figures, in the deck's existing visual language.

Every drawn line is asserted to be a walk over adjacent cells, and every panel
is asserted to sit inside the canvas - an illustration that cheats visually
would undercut the point it is making.
"""
import random

OUT    = '/data/dante_data/documents/Masters/courses/Thesis/g-medmamba/doc/figures'
INK    = '#1c2430'
MUTED  = '#7b8794'
WALL   = '#5a6472'
PURPLE = '#6a5a92'
GREEN  = '#4e8c6a'
RED    = '#b4553f'
PAPER  = '#fcfbf8'
ERASED = '#c3c7cd'

STYLE = """<style>
  .t  {font-size:15px;fill:#1c2430;text-anchor:middle}
  .tb {font-size:15.5px;font-weight:600;fill:#1c2430;text-anchor:middle}
  .ts {font-size:12.5px;fill:#7b8794;text-anchor:middle}
  .ti {font-size:21px;font-weight:600;fill:#1c2430}
  .tsub{font-size:14px;fill:#7b8794}
  .cap{font-size:13.5px;fill:#7b8794}
  .cp {stroke:#6a5a92;stroke-width:2;fill:none;marker-end:url(#arp)}
</style>
"""

def head(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">\n'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>\n'
            f'<defs><marker id="arp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{PURPLE}"/>'
            f'</marker></defs>\n' + STYLE)


def wrap(x, y, text, width_px, size=13.5, cls='cap', lead=20):
    """Left-aligned wrapped text. `text` may contain <tspan> spans as atoms."""
    import re
    atoms = re.findall(r'<tspan.*?</tspan>|\S+', text)
    per = int(width_px / (size * 0.47))
    lines, cur, n = [], [], 0
    for a in atoms:
        vis = re.sub(r'<.*?>', '', a)
        if n + len(vis) + 1 > per and cur:
            lines.append(' '.join(cur)); cur, n = [], 0
        cur.append(a); n += len(vis) + 1
    if cur: lines.append(' '.join(cur))
    out = [f'<text x="{x}" y="{y}" class="{cls}">']
    for i, ln in enumerate(lines):
        out.append(f'<tspan x="{x}" dy="{0 if i == 0 else lead}">{ln}</tspan>')
    out.append('</text>')
    return '\n'.join(out), y + lead * (len(lines) - 1)


# ---------------------------------------------------------------- maze model
def gen_maze(n, seed):
    rnd = random.Random(seed)
    walls = set()
    for r in range(n):
        for c in range(n):
            if r + 1 < n: walls.add(((r, c), (r + 1, c)))
            if c + 1 < n: walls.add(((r, c), (r, c + 1)))
    seen = [[False] * n for _ in range(n)]
    parent, stack = {}, [(0, 0)]
    seen[0][0] = True
    while stack:
        r, c = stack[-1]
        nb = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= r + dr < n and 0 <= c + dc < n and not seen[r + dr][c + dc]]
        if not nb:
            stack.pop(); continue
        q = rnd.choice(nb)
        walls.discard(((r, c), q)); walls.discard((q, (r, c)))
        parent[q] = (r, c); seen[q[0]][q[1]] = True
        stack.append(q)
    path, cur = [], (n - 1, n - 1)
    while cur != (0, 0):
        path.append(cur); cur = parent[cur]
    path.append((0, 0)); path.reverse()
    return walls, path


def nbrs(cell, walls, n):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= r + dr < n and 0 <= c + dc < n
            and ((r, c), (r + dr, c + dc)) not in walls
            and ((r + dr, c + dc), (r, c)) not in walls]


def stub(path, walls, n, at, want, avoid=()):
    """A short dead-end detour of about `want` cells leaving path[at]."""
    block = set(path) | set(avoid)
    best = []
    def walk(cell, acc):
        nonlocal best
        if len(acc) >= want:
            if not best or abs(len(acc) - want) < abs(len(best) - want):
                best = list(acc)
            return
        opts = [q for q in nbrs(cell, walls, n) if q not in block and q not in acc]
        if not opts:
            if not best or abs(len(acc) - want) < abs(len(best) - want):
                best = list(acc)
            return
        for q in opts:
            walk(q, acc + [q])
    for q in nbrs(path[at], walls, n):
        if q not in block:
            walk(q, [q])
    return [path[at]] + best


def adjacent(cells):
    return all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(cells, cells[1:]))


def attempt(path, k, det):
    """True path up to index k, then a wrong detour. Asserted connected."""
    out = path[:k] + det
    assert adjacent(out), 'illustration would draw a jump between non-adjacent cells'
    return out


def pick(n, lo, hi):
    """First seed whose solution is legible and that offers two short detours
    - one early, one late - each a visible dead-end stub."""
    for seed in range(600):
        walls, path = gen_maze(n, seed)
        L = len(path)
        if not (lo <= L <= hi):
            continue
        early = [(i, stub(path, walls, n, i, 4)) for i in range(2, int(L * 0.45))]
        early = [(i, d) for i, d in early if len(d) >= 4]
        if not early:
            continue
        k1, d1 = max(early, key=lambda t: len(t[1]))
        late = [(i, stub(path, walls, n, i, 4, avoid=d1)) for i in range(int(L * 0.55), L - 3)]
        late = [(i, d) for i, d in late if len(d) >= 4]
        if not late:
            continue
        k2, d2 = max(late, key=lambda t: len(t[1]))
        return walls, path, k1, d1, k2, d2
    raise SystemExit('no suitable maze found')


def draw_maze(x0, y0, cell, walls, n, lines, title=None, sub=None):
    s = [f'<rect x="{x0-6}" y="{y0-6}" width="{n*cell+12}" height="{n*cell+12}" fill="{PAPER}" '
         f'stroke="#e6e3db" stroke-width="1.4" rx="5"/>']
    s.append(f'<path d="M{x0+cell} {y0} L{x0+n*cell} {y0} L{x0+n*cell} {y0+(n-1)*cell}" '
             f'stroke="{WALL}" stroke-width="2.2" fill="none"/>')
    s.append(f'<path d="M{x0} {y0} L{x0} {y0+n*cell} L{x0+(n-1)*cell} {y0+n*cell}" '
             f'stroke="{WALL}" stroke-width="2.2" fill="none"/>')
    for a, b in sorted(walls):
        (r1, c1), (r2, _) = a, b
        if r2 == r1 + 1:
            X, Y = x0 + c1 * cell, y0 + (r1 + 1) * cell
            s.append(f'<line x1="{X}" y1="{Y}" x2="{X+cell}" y2="{Y}" stroke="{WALL}" stroke-width="1.5"/>')
        else:
            X, Y = x0 + (c1 + 1) * cell, y0 + r1 * cell
            s.append(f'<line x1="{X}" y1="{Y}" x2="{X}" y2="{Y+cell}" stroke="{WALL}" stroke-width="1.5"/>')
    for cells, col, wdt, dash, op in lines:
        assert adjacent(cells), 'non-adjacent cells in a drawn line'
        pts = ' '.join(f'{x0+c*cell+cell/2:.1f},{y0+r*cell+cell/2:.1f}' for r, c in cells)
        da = f' stroke-dasharray="{dash}"' if dash else ''
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wdt}" '
                 f'stroke-linecap="round" stroke-linejoin="round" opacity="{op}"{da}/>')
    s.append(f'<circle cx="{x0+cell/2}" cy="{y0+cell/2}" r="{cell*0.20}" fill="{GREEN}"/>')
    s.append(f'<circle cx="{x0+(n-1)*cell+cell/2}" cy="{y0+(n-1)*cell+cell/2}" r="{cell*0.20}" fill="{RED}"/>')
    if title:
        s.append(f'<text x="{x0+n*cell/2}" y="{y0-20}" class="tb">{title}</text>')
    if sub:
        s.append(f'<text x="{x0+n*cell/2}" y="{y0+n*cell+28}" class="ts">{sub}</text>')
    return '\n'.join(s)


# ================================================================== fig 19
N = 8
walls, path, k1, d1, k2, d2 = pick(N, 18, 30)
try1 = attempt(path, k1, d1)
try2 = attempt(path, k2, d2)

W, H, cell = 1040, 470, 28
mz = N * cell                                   # 240
xs = [50, 408, 766]
assert xs[-1] + mz + 6 <= W, 'panels overflow the canvas'
top = 112

g = [head(W, H)]
g.append('<text x="34" y="40" class="ti">TRM, as solving a maze in pencil</text>')
g.append('<text x="34" y="65" class="tsub">One person, one pencil, one eraser — used over and over.</text>')

g.append(draw_maze(xs[0], top, cell, walls, N, [], 'x — the maze',
                   'looked at once. never changes.'))
g.append(draw_maze(xs[1], top, cell, walls, N,
                   [(d1, ERASED, 4.5, '1 6', 0.9), (try2, PURPLE, 4.5, None, 0.95)],
                   'y — the path so far', 'pencil. rubbed out, redrawn.'))
g.append(draw_maze(xs[2], top, cell, walls, N, [(path, GREEN, 4.5, None, 0.95)],
                   'y — when it is done', 'the same line, a few passes later.'))

# f, sitting between the maze and the pencil line
fx, fy = 341, top + mz / 2
g.append(f'<circle cx="{fx}" cy="{fy}" r="27" fill="#efecf6" stroke="{PURPLE}" stroke-width="1.8"/>')
g.append(f'<text x="{fx}" y="{fy+6}" class="tb" fill="{PURPLE}">f</text>')
g.append(f'<text x="{fx}" y="{fy+52}" class="ts">you</text>')
g.append(f'<text x="{fx}" y="{fy+68}" class="ts">one brain</text>')
g.append(f'<text x="{fx}" y="{fy+84}" class="ts">21 passes</text>')
g.append(f'<path d="M{xs[0]+mz+10},{fy} L{fx-32},{fy}" class="cp"/>')
g.append(f'<path d="M{fx+32},{fy} L{xs[1]-12},{fy}" class="cp"/>')
g.append(f'<path d="M{xs[1]+mz+10},{fy} L{xs[2]-12},{fy}" class="cp"/>')
g.append(f'<text x="{(xs[1]+mz+xs[2])/2}" y="{fy-12}" class="ts">repeat</text>')

t, _ = wrap(58, 418,
            f'<tspan font-weight="600" fill="{INK}">Six scribbles of thinking, then one stroke of the pencil.</tspan> '
            f'That is one round. The scribbles are '
            f'<tspan font-weight="600" fill="{INK}">z</tspan> — dead ends you now remember, bits you are unsure of — '
            f'and they are never shown to anyone. The pencil line is '
            f'<tspan font-weight="600" fill="{INK}">y</tspan>, and it is the only thing anyone ever sees. '
            f'<tspan fill="{RED}">Illustration of the idea, not model output.</tspan>', 930)
g.append(t)
g.append('</svg>')
open(f'{OUT}/fig19_trm_maze.svg', 'w').write('\n'.join(g))

# ================================================================== fig 20
W2, H2, c2 = 1040, 452, 26
mz2 = N * c2                                    # 208
xs2 = [46, 292, 538, 784]
assert xs2[-1] + mz2 + 6 <= W2, 'panels overflow the canvas'
top2 = 116

h = [head(W2, H2)]
h.append('<text x="34" y="40" class="ti">Why it improves: rub a bit out, redraw a bit</text>')
h.append('<text x="34" y="65" class="tsub">The answer is never thrown away and restarted. Each pass edits the line already on the paper.</text>')

panels = [
    ([(try1, PURPLE, 4, None, 0.95)],
     'pass 1', 'a confident start,', 'straight into a dead end'),
    ([(d1, ERASED, 4, '1 5', 0.95), (try2, PURPLE, 4, None, 0.95)],
     'pass 2', 'that stretch rubbed out,', 'the next one still wrong'),
    ([(d1 + [], ERASED, 4, '1 5', 0.6), (d2, ERASED, 4, '1 5', 0.95),
      (path[:len(path) - 2], PURPLE, 4, None, 0.95)],
     'pass 3', 'both detours gone,', 'not yet at the exit'),
    ([(path, GREEN, 4, None, 0.95)],
     'pass 4', 'the line reaches', 'the end'),
]
for x, (lines, lab, s1, s2) in zip(xs2, panels):
    h.append(draw_maze(x, top2, c2, walls, N, lines, lab, s1))
    h.append(f'<text x="{x+mz2/2}" y="{top2+mz2+45}" class="ts">{s2}</text>')

t2, y2 = wrap(46, 400,
              f'Grey dashes are what was rubbed out. '
              f'<tspan font-weight="600" fill="{INK}">This is why the model can be tiny:</tspan> '
              f'it never has to be right in one shot, so it never needs the capacity to be. '
              f'Every pass is the same two layers doing the same job — take an imperfect line, hand back a better one.', 940)
h.append(t2)
h.append(f'<text x="46" y="{y2+20}" class="cap" fill="{RED}">Illustration of the idea, not model output.</text>')
h.append('</svg>')
open(f'{OUT}/fig20_trm_maze_refine.svg', 'w').write('\n'.join(h))

print(f'ok  maze {N}x{N}  solution={len(path)}  detours={len(d1)},{len(d2)} at {k1},{k2}')
