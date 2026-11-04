"""Draws the project's logo, banner and README diagrams as self-contained SVG files.

Everything is pixels on a grid, including the type, so the files need no fonts and look the same
in a README, where SVGs run as images. Animation is CSS and SMIL only, because GitHub strips
scripts, and it switches off for readers who ask for reduced motion.

    python3 brand/build.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
FELT, RAISED, LINE = "#17235e", "#22307a", "#3a4a9a"
IVORY, GOLD, DARK_GOLD, RED, MUTED, INK = "#fbf6e9", "#f2c14e", "#a97f25", "#e5484d", "#aab4e6", "#0e1640"
QUIET = "@media(prefers-reduced-motion:reduce){*{animation:none!important}.packet{display:none}}"

GLYPHS = {
    "A": (14, 17, 17, 31, 17, 17, 17), "B": (30, 17, 17, 30, 17, 17, 30), "C": (14, 17, 16, 16, 16, 17, 14),
    "D": (30, 17, 17, 17, 17, 17, 30), "E": (31, 16, 16, 30, 16, 16, 31), "F": (31, 16, 16, 30, 16, 16, 16),
    "G": (14, 17, 16, 23, 17, 17, 15), "H": (17, 17, 17, 31, 17, 17, 17), "I": (14, 4, 4, 4, 4, 4, 14),
    "J": (7, 2, 2, 2, 2, 18, 12), "K": (17, 18, 20, 24, 20, 18, 17), "L": (16, 16, 16, 16, 16, 16, 31),
    "M": (17, 27, 21, 21, 17, 17, 17), "N": (17, 17, 25, 21, 19, 17, 17), "O": (14, 17, 17, 17, 17, 17, 14),
    "P": (30, 17, 17, 30, 16, 16, 16), "Q": (14, 17, 17, 17, 21, 18, 13), "R": (30, 17, 17, 30, 20, 18, 17),
    "S": (15, 16, 16, 14, 1, 1, 30), "T": (31, 4, 4, 4, 4, 4, 4), "U": (17, 17, 17, 17, 17, 17, 14),
    "V": (17, 17, 17, 17, 17, 10, 4), "W": (17, 17, 17, 21, 21, 21, 10), "X": (17, 17, 10, 4, 10, 17, 17),
    "Y": (17, 17, 17, 10, 4, 4, 4), "Z": (31, 1, 2, 4, 8, 16, 31),
    "a": (0, 0, 14, 1, 15, 17, 15), "b": (16, 16, 22, 25, 17, 17, 30), "c": (0, 0, 14, 16, 16, 17, 14),
    "d": (1, 1, 13, 19, 17, 17, 15), "e": (0, 0, 14, 17, 31, 16, 14), "f": (6, 9, 8, 28, 8, 8, 8),
    "g": (0, 0, 15, 17, 17, 15, 1, 1, 14), "h": (16, 16, 22, 25, 17, 17, 17), "i": (4, 0, 12, 4, 4, 4, 14),
    "j": (2, 0, 6, 2, 2, 2, 2, 18, 12), "k": (16, 16, 18, 20, 24, 20, 18), "l": (12, 4, 4, 4, 4, 4, 14),
    "m": (0, 0, 26, 21, 21, 17, 17), "n": (0, 0, 22, 25, 17, 17, 17), "o": (0, 0, 14, 17, 17, 17, 14),
    "p": (0, 0, 30, 17, 17, 30, 16, 16, 16), "q": (0, 0, 13, 19, 17, 15, 1, 1, 1), "r": (0, 0, 22, 25, 16, 16, 16),
    "s": (0, 0, 14, 16, 14, 1, 30), "t": (8, 8, 28, 8, 8, 9, 6), "u": (0, 0, 17, 17, 17, 19, 13),
    "v": (0, 0, 17, 17, 17, 10, 4), "w": (0, 0, 17, 17, 21, 21, 10), "x": (0, 0, 17, 10, 4, 10, 17),
    "y": (0, 0, 17, 17, 17, 15, 1, 1, 14), "z": (0, 0, 31, 2, 4, 8, 31),
    "0": (14, 17, 19, 21, 25, 17, 14), "1": (4, 12, 4, 4, 4, 4, 14), "2": (14, 17, 1, 2, 4, 8, 31),
    "3": (31, 2, 4, 2, 1, 17, 14), "4": (2, 6, 10, 18, 31, 2, 2), "5": (31, 16, 30, 1, 1, 17, 14),
    "6": (6, 8, 16, 30, 17, 17, 14), "7": (31, 1, 2, 4, 8, 8, 8), "8": (14, 17, 17, 14, 17, 17, 14),
    "9": (14, 17, 17, 15, 1, 2, 12),
    ".": (0, 0, 0, 0, 0, 12, 12), ",": (0, 0, 0, 0, 12, 4, 8), ":": (0, 12, 12, 0, 12, 12, 0),
    "-": (0, 0, 0, 31, 0, 0, 0), "+": (0, 4, 4, 31, 4, 4, 0), "/": (1, 1, 2, 4, 8, 16, 16),
    "_": (0, 0, 0, 0, 0, 0, 31), "=": (0, 0, 31, 0, 31, 0, 0), "?": (14, 17, 1, 2, 4, 0, 4),
    "*": (0, 4, 21, 14, 21, 4, 0), "!": (4, 4, 4, 4, 4, 0, 4), "'": (4, 4, 8, 0, 0, 0, 0), '"': (10, 10, 10, 0, 0, 0, 0),
    "(": (2, 4, 8, 8, 8, 4, 2), ")": (8, 4, 2, 2, 2, 4, 8), "[": (14, 8, 8, 8, 8, 8, 14),
    "]": (14, 2, 2, 2, 2, 2, 14), "{": (6, 8, 8, 16, 8, 8, 6), "}": (12, 2, 2, 1, 2, 2, 12),
    ">": (8, 4, 2, 1, 2, 4, 8), "<": (2, 4, 8, 16, 8, 4, 2), " ": (0, 0, 0, 0, 0, 0, 0),
}

# A king's card that is also a bot. Eyes are drawn apart so they can blink.
MARK = """
..G....G....G..
..G...GGG...G..
..GG..GGG..GG..
..GGGGGGGGGGG..
..GGRGGGGGRGG..
..ggggggggggg..
.NNNNNNNNNNNNN.
.NIIIIIIIIIIIN.
.NIIEEIIIEEIIN.
.NIIEEIIIEEIIN.
.NIIIIIIIIIIIN.
.NIIIIIRIIIIIN.
.NIIIIRRRIIIIN.
.NIIIRRRRRIIIN.
.NIIIIRRRIIIIN.
.NIIIIIRIIIIIN.
.NIIIIIIIIIIIN.
.NIMIMIMIMIMIN.
.NIIIIIIIIIIIN.
.NNNNNNNNNNNNN.
""".split()
MARK_COLORS = {"G": GOLD, "g": DARK_GOLD, "R": RED, "N": INK, "I": IVORY, "M": INK}
PIPS = {
    "club": (["..###..", "..###..", "##.#.##", "#######", "##.#.##", "...#...", "..###.."], INK),
    "diamond": (["...#...", "..###..", ".#####.", "#######", ".#####.", "..###..", "...#..."], RED),
    "heart": ([".##.##.", "#######", "#######", ".#####.", "..###..", "...#...", "......."], RED),
    "spade": (["...#...", "..###..", ".#####.", "#######", "##.#.##", "...#...", "..###.."], INK),
}


def runs(rows: list[str], x: float, y: float, s: float, on: str) -> str:
    """A path through the horizontal runs of the cells marked with a character in ``on``."""
    out = []
    for j, row in enumerate(rows):
        i = 0
        while i < len(row):
            if row[i] in on:
                k = i
                while k < len(row) and row[k] in on:
                    k += 1
                out.append(f"M{x + i * s:g} {y + j * s:g}h{(k - i) * s:g}v{s:g}h{-(k - i) * s:g}z")
                i = k
            else:
                i += 1
    return "".join(out)


def text_path(text: str, x: float, y: float, s: float) -> str:
    out = []
    for n, ch in enumerate(text):
        rows = [f"{bits:05b}" for bits in GLYPHS.get(ch, GLYPHS["?"])]
        out.append(runs(rows, x + n * 6 * s, y, s, "1"))
    return "".join(out)


def width(text: str, s: float) -> float:
    return (len(text) * 6 - 1) * s


def text(value: str, x: float, y: float, s: float = 2, fill: str = IVORY, anchor: str = "start") -> str:
    if anchor == "middle":
        x -= width(value, s) / 2
    elif anchor == "end":
        x -= width(value, s)
    return f'<path fill="{fill}" d="{text_path(value, x, y, s)}"/>'


def mark(x: float, y: float, s: float) -> str:
    parts = [f'<path fill="{c}" d="{runs(MARK, x, y, s, k)}"/>' for k, c in MARK_COLORS.items()]
    parts.append(f'<path class="eyes" fill="{INK}" d="{runs(MARK, x, y, s, "E")}"/>')
    # What shows when the eyes close.
    return f'<path fill="{IVORY}" d="{runs(MARK, x, y, s, "E")}"/>' + "".join(parts)


def svg(w: int, h: int, title: str, desc: str, style: str, body: str, background: bool = True) -> str:
    base = f'.eyes{{animation:blink 5s steps(1) infinite}}@keyframes blink{{0%,93%{{opacity:1}}94%,97%{{opacity:0}}98%{{opacity:1}}}}{style}{QUIET}'
    back = f'<rect width="{w}" height="{h}" fill="{FELT}"/>' if background else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" shape-rendering="crispEdges">'
            f"<title>{title}</title><desc>{desc}</desc><style>{base}</style>{back}{body}</svg>\n")


def card(x: float, y: float, s: float, pip: str) -> str:
    rows, color = PIPS[pip]
    w, h = 13 * s, 17 * s
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{INK}"/>'
            f'<rect x="{x + s}" y="{y + s}" width="{w - 2 * s}" height="{h - 2 * s}" fill="{IVORY}"/>'
            f'<path fill="{color}" d="{runs(rows, x + 3 * s, y + 5 * s, s, "#")}"/>')


def banner() -> str:
    s, w, h = 10, 1280, 340
    word1, word2 = "ESTIMATION", "ARENA"
    x0, y1, y2 = 270, 58, 148
    letters = text_path(word1, x0, y1, s) + text_path(word2, x0, y2, s)
    shadow = text_path(word1, x0 + s // 2 * 1, y1 + s // 2, s) + text_path(word2, x0 + s // 2, y2 + s // 2, s)
    cards = []
    for k, pip in enumerate(("club", "diamond", "heart", "spade")):
        x = 880 + k * 88
        cards.append(f'<g class="deal" style="animation-delay:{0.25 * k:g}s">{card(x, 170, 6, pip)}</g>')
    dots = "".join(f"M{x} {y}h4v4h-4z" for y in range(14, h, 28) for x in range(14 + (y // 28 % 2) * 14, w, 28))
    style = (".deal{animation:deal .7s steps(10) both}@keyframes deal{from{transform:translateX(420px);opacity:0}to{transform:none;opacity:1}}"
             ".shine{animation:shine 7s steps(46) infinite}@keyframes shine{0%{transform:translateX(0)}35%,100%{transform:translateX(1000px)}}")
    body = (f'<path fill="{RAISED}" d="{dots}"/>'
            f'<rect x="8" y="8" width="{w - 16}" height="{h - 16}" fill="none" stroke="{LINE}" stroke-width="4"/>'
            f"{mark(70, 60, 11)}"
            f'<clipPath id="word"><path d="{letters}"/></clipPath>'
            f'<path fill="{DARK_GOLD}" d="{shadow}"/>'
            f'<path fill="{IVORY}" d="{text_path(word1, x0, y1, s)}"/><path fill="{GOLD}" d="{text_path(word2, x0, y2, s)}"/>'
            f'<g clip-path="url(#word)"><rect class="shine" x="{x0 - 120}" y="40" width="60" height="200" fill="#fff" opacity=".75"/></g>'
            f'{text("Write a bot. Get an Elo.", x0, 254, 4, MUTED)}'
            + "".join(cards))
    return svg(w, h, "Estimation Arena", "A pixel-art king's card with a robot face, beside the words Estimation Arena. Write a bot. Get an Elo.", style, body)


def logo_mark() -> str:
    body = mark(0, 0, 8)
    return svg(120, 160, "Estimation Arena", "A pixel-art king's card with a robot face", "", body, background=False)


def favicon() -> str:
    rows = "".join(f'<path fill="{c}" d="{runs(MARK, 2, 0, 1, k)}"/>' for k, c in {**MARK_COLORS, "E": INK}.items())
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" shape-rendering="crispEdges"><title>Estimation Arena</title>{rows}</svg>\n')


# ---- diagrams
def node(x, y, w, h, title, lines=(), accent=False, klass="") -> str:
    edge = GOLD if accent else LINE
    top = y + (h - (21 + 22 * len(lines) + (4 if lines else 0))) / 2
    out = [f'<g class="{klass}">' if klass else "<g>",
           f'<path fill="{edge}" d="M{x + 4} {y}h{w - 8}v4h4v{h - 8}h-4v4h-{w - 8}v-4h-4v-{h - 8}h4z"/>',
           f'<path fill="{RAISED}" d="M{x + 4} {y + 4}h{w - 8}v{h - 8}h-{w - 8}z"/>',
           text(title, x + w / 2, top, 3, GOLD if accent else IVORY, "middle")]
    for k, line in enumerate(lines):
        out.append(text(line, x + w / 2, top + 31 + 22 * k, 2, MUTED, "middle"))
    out.append("</g>")
    return "".join(out)


def wire(points, color=LINE, head=True, tail=False) -> str:
    """An orthogonal line of 4px, with a pixel arrowhead at the end and optionally the start."""
    d = "M" + "L".join(f"{x} {y}" for x, y in points)
    out = [f'<path d="{d}" fill="none" stroke="{color}" stroke-width="4"/>']
    if head:
        out.append(_head(points[-2], points[-1], color))
    if tail:
        out.append(_head(points[1], points[0], color))
    return "".join(out)


def _head(a, b, color) -> str:
    x, y = b
    dx = (b[0] > a[0]) - (b[0] < a[0])
    dy = (b[1] > a[1]) - (b[1] < a[1])
    cells = []
    for k in range(1, 4):
        cx, cy = x - dx * k * 4, y - dy * k * 4
        w = k * 2 - 1
        for j in range(-(w // 2), w // 2 + 1):
            cells.append((cx + dy * j * 4 - 2, cy + dx * j * 4 - 2))
    return f'<path fill="{color}" d="' + "".join(f"M{a} {c}h4v4h-4z" for a, c in cells) + '"/>'


def packet(points, dur: float, begin: float = 0, cycle: float | None = None, color=GOLD, size=10) -> str:
    """A pixel that runs along a line, then waits so several can take turns in one ``cycle``."""
    d = "M" + "L".join(f"{x} {y}" for x, y in points)
    cycle = cycle or dur
    a, b = begin / cycle, (begin + dur) / cycle
    keys = ";".join(f"{v:.4f}" for v in sorted({0.0, a, b, 1.0}))
    points_at = {0.0: 0, a: 0, b: 1, 1.0: 1}
    motion = ";".join(str(points_at[v]) for v in sorted({0.0, a, b, 1.0}))
    show = ";".join("1" if a <= v < b else "0" for v in sorted({0.0, a, b, 1.0}))
    return (f'<rect class="packet" x="{-size / 2}" y="{-size / 2}" width="{size}" height="{size}" fill="{color}" opacity="0">'
            f'<animateMotion path="{d}" dur="{cycle}s" repeatCount="indefinite" keyPoints="{motion}" keyTimes="{keys}" calcMode="linear"/>'
            f'<animate attributeName="opacity" values="{show}" keyTimes="{keys}" dur="{cycle}s" repeatCount="indefinite" calcMode="discrete"/></rect>')


def how_it_works() -> str:
    w, h = 1000, 420
    bot, engine, ana = (30, 60, 220, 100), (720, 60, 250, 100), (30, 280, 230, 100)
    body = [
        wire([(250, 110), (390, 110)], tail=True), wire([(600, 110), (720, 110)], tail=True),
        text("JSON in", 320, 86, 2, MUTED, "middle"), text("action out", 320, 124, 2, MUTED, "middle"),
        text("rules", 660, 86, 2, MUTED, "middle"),
        wire([(140, 280), (140, 225), (440, 225), (440, 160)]),
        text("fixed bots to measure against", 150, 200, 2, MUTED),
        wire([(530, 160), (530, 280)]), wire([(620, 330), (650, 330)]), wire([(780, 330), (810, 330)]),
        node(*bot, "YOUR BOT", ["bots/NAME/"], klass="n"),
        node(390, 60, 210, 100, "HARNESS", ["arena/", "runs every game"], accent=True),
        node(*engine, "RULES ENGINE", ["estimation-engine"]),
        node(*ana, "ANCHORS", ["Random to Hybrid"]),
        node(430, 280, 190, 100, "RESULTS", ["results/*.json"]),
        node(650, 280, 130, 100, "ELO FIT", ["all games"]),
        node(810, 280, 160, 100, "PAGE", ["leaderboard"], accent=True),
        packet([(250, 110), (390, 110)], .7, 0, 5), packet([(390, 110), (250, 110)], .7, .8, 5),
        packet([(600, 110), (720, 110)], .7, 1.6, 5), packet([(140, 280), (140, 225), (440, 225), (440, 160)], 1.1, 2.4, 5),
        packet([(530, 160), (530, 280)], .7, 3.5, 5), packet([(620, 330), (650, 330), (780, 330), (810, 330)], .7, 4.2, 5),
    ]
    return svg(w, h, "How it works", "Your bot exchanges JSON and actions with the harness. The harness uses the rules engine and plays the anchors. Results feed an Elo fit, which feeds the leaderboard page.", "", "".join(body))


def one_decision() -> str:
    w, h = 1000, 520
    a, b = 170, 830
    head_h = 96
    msgs = [(148, a, b, '{"type":"hello","seat_seed":...,"rules":...}'),
            (200, a, b, '{"type":"act","obs":{...},"legal":[68,70,81]}'),
            (252, b, a, "70"),
            (340, a, b, '{"type":"bye"}')]
    body = [node(a - 110, 14, 220, head_h, "HARNESS", ["arena/"], accent=True),
            node(b - 110, 14, 220, head_h, "YOUR BOT", ["fresh process", "per game"])]
    for x in (a, b):
        body.append(f'<path fill="{LINE}" d="' + "".join(f"M{x - 2} {y}h4v8h-4z" for y in range(112, 360, 16)) + '"/>')
    for y, fro, to, label in msgs:
        body.append(wire([(fro, y), (to, y)], GOLD if label == "70" else LINE))
        body.append(text(label, (a + b) / 2, y - 22, 2, GOLD if label == "70" else IVORY, "middle"))
    body.append(f'<rect x="{a + 90}" y="272" width="{b - a - 180}" height="30" fill="{RAISED}"/>')
    body.append(text("illegal id or over 5 s = forfeit", (a + b) / 2, 280, 2, "#f3a6a8", "middle"))
    for k, (y, fro, to, _) in enumerate(msgs):
        body.append(packet([(fro, y), (to, y)], 1, k * 1.2, 6, RED if to == a else GOLD))
    cells = [("DASH", "0-1", 110), ("BIDS", "2-51", 150), ("WITH", "52", 110), ("PASS", "53", 110), ("ESTIMATES", "54-67", 200), ("CARDS", "68-119", 180)]
    body.append(text("Action ids", 40, 374, 2, MUTED))
    x = 40
    for name, ids, cw in cells:
        body.append(node(x, 400, cw, 92, name, [ids]))
        x += cw + 8
    return svg(w, h, "One decision", "The harness says hello, sends an observation and the legal action ids, and the bot answers with one id. An illegal id or taking over five seconds is a forfeit. Then the harness says bye. Action ids: 0 to 1 dash, 2 to 51 bids, 52 with, 53 pass, 54 to 67 estimates, 68 to 119 cards.", "", "".join(body))


def two_doors() -> str:
    w, h = 1000, 380
    panel = lambda x, wd, label: (f'<rect x="{x}" y="30" width="{wd}" height="260" fill="none" stroke="{LINE}" stroke-width="4" stroke-dasharray="8 8"/>'
                                  f'<rect x="{x + 16}" y="20" width="{width(label, 3) + 16}" height="24" fill="{FELT}"/>{text(label, x + 24, 22, 3, GOLD)}')
    body = [panel(30, 590, "COMPETE"), panel(650, 320, "TRAIN"),
            wire([(290, 165), (320, 165), (320, 115), (345, 115)]), wire([(320, 165), (320, 215), (345, 215)]),
            wire([(810, 150), (810, 180)]),
            wire([(810, 270), (810, 335), (170, 335), (170, 215)], GOLD),
            text("wrap it", 490, 348, 2, GOLD, "middle"),
            node(50, 115, 240, 100, "MANIFEST", ["bots/NAME/", "manifest.json"], accent=True),
            node(345, 70, 255, 90, "WIRE TIER", ["any language"]),
            node(345, 170, 255, 90, "PYTHON TIER", ["make(seed, rules)"]),
            node(670, 60, 280, 90, "ENVIRONMENT", ["arena/env.py", "PettingZoo AEC"]),
            node(670, 180, 280, 90, "YOUR POLICY", ["your own training"]),
            packet([(810, 270), (810, 335), (170, 335), (170, 215)], 2.2, 0, 5), packet([(290, 165), (320, 165), (320, 115), (345, 115)], .7, 2.4, 5),
            packet([(290, 165), (320, 165), (320, 215), (345, 215)], .7, 3.3, 5)]
    return svg(w, h, "Two doors", "To compete, a manifest declares a wire bot in any language or a Python function. To train, use the PettingZoo environment in arena/env.py with your own policy, then wrap the policy in a manifest.", "", "".join(body))


def submit() -> str:
    w, h = 1000, 190
    nodes = [("PR", ["your pull", "request"]), ("SMOKE", ["2 sets vs", "Random and", "Heuristic"]), ("COMMENT", ["the result", "on your PR"]),
             ("MERGE", ["joins the", "ladder"]), ("NIGHTLY", ["rated on a", "secret seed"]), ("PAGE", ["leaderboard", "updates"])]
    cw, gap, x0, y, nh = 140, 26, 18, 30, 130
    body, styles = [], []
    for k, (title, lines) in enumerate(nodes):
        x = x0 + k * (cw + gap)
        if k:
            body.append(wire([(x - gap, y + nh // 2), (x, y + nh // 2)]))
        body.append(node(x, y, cw, nh, title, lines, accent=k == len(nodes) - 1))
        body.append(f'<rect class="lit l{k}" x="{x}" y="{y + nh + 8}" width="{cw}" height="6" fill="{GOLD}"/>')
        styles.append(f".l{k}{{animation-delay:{k * 1.0}s}}")
    style = ".lit{opacity:.15;animation:lit 6s steps(1) infinite}@keyframes lit{0%,16%{opacity:1}17%,100%{opacity:.15}}" + "".join(styles)
    return svg(w, h, "Submit a bot", "Open a pull request. A smoke test plays 2 sets against Random and Heuristic and comments the result. After merging, the nightly ladder rates the bot on a secret seed and the page updates.", style, "".join(body))


def main() -> None:
    files = {"banner.svg": banner(), "mark.svg": logo_mark(), "favicon.svg": favicon(), "how-it-works.svg": how_it_works(),
             "one-decision.svg": one_decision(), "two-doors.svg": two_doors(), "submit.svg": submit()}
    for name, content in files.items():
        (HERE / name).write_text(content)
        if name in ("mark.svg", "favicon.svg"):
            (HERE.parent / "arena" / "static" / name).write_text(content)
        print(f"{name}: {len(content):,} bytes")


if __name__ == "__main__":
    main()
