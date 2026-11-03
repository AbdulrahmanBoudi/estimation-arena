"""Build the leaderboard from ``results/*.json`` and ``ladder/pairs.json``.

    uv run python -m arena.leaderboard

One Bradley-Terry fit over every pairwise set result (the imported ladder plus every
submission's runs), RandomBot fixed at 400, gives every bot one Elo on the same scale.
Writes ``site/leaderboard.json`` and ``site/index.html``, a single self-contained page.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from estimation.arena.ratings import ANCHOR, ANCHOR_RATING, DEFAULT_PRIOR, PairRecord, bradley_terry

from .bots import ROOT, SUB_PREFIX, load_manifest
from .import_ladder import LADDER_FILE
from .run import ANCHORS_FILE, RESULTS_DIR

SITE_DIR = ROOT / "site"


def _pairs(records: list[dict[str, Any]]) -> list[PairRecord]:
    return [
        PairRecord(str(p["a"]), str(p["b"]), int(p["a_wins"]), int(p["draws"]), int(p["b_wins"]))
        for p in records
    ]


def build() -> dict[str, Any]:
    anchors: list[dict[str, str]] = json.loads(ANCHORS_FILE.read_text())
    ladder = json.loads(LADDER_FILE.read_text()) if LADDER_FILE.is_file() else {"pairs": []}
    results = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    pairs = _pairs(ladder["pairs"])
    for r in results:
        pairs += _pairs(r["pairs"])
    ratings = bradley_terry(pairs, ANCHOR, ANCHOR_RATING, DEFAULT_PRIOR) if pairs else {}

    rows: list[dict[str, Any]] = []
    for r in results:
        key = SUB_PREFIX + r["bot"]
        try:
            current = load_manifest(r["bot"]).to_dict()
        except ValueError:
            current = r["manifest"]
        timings = [v["timing"] for v in r["versus"]]
        decisions = sum(t["count"] for t in timings)
        rows.append(
            {
                "name": r["bot"],
                "author": r["manifest"].get("author", ""),
                "kind": r["manifest"].get("kind", ""),
                "description": current.get("description", r["manifest"].get("description", "")),
                "key": key,
                "method": current.get("method", []),
                "elo": ratings.get(key),
                "games": sum(v["games"] for v in r["versus"]),
                "sets": r["sets"],
                "seed": r["seed"],
                "played_at": r["played_at"],
                "engine": r.get("engine", ""),
                "p95_ms": max((t["p95_s"] for t in timings), default=0.0) * 1000,
                "over_budget": (sum(t["over_budget"] for t in timings) / decisions) if decisions else 0.0,
                "versus": [
                    {
                        "anchor": v["anchor"],
                        "rsa": v["rsa"]["mean"],
                        "rsa_low": v["rsa"]["low"],
                        "rsa_high": v["rsa"]["high"],
                        "first_place": v["first_place"]["mean"],
                    }
                    for v in r["versus"]
                ],
                "anchor": False,
            }
        )
    for a in anchors:
        rows.append(
            {
                "name": a["name"],
                "author": a.get("author", "Yosif Elessawi"),
                "kind": "anchor",
                "description": a["spec"],
                "key": a["spec"],
                "method": a.get("method", []),
                "elo": ratings.get(a["spec"]),
                "games": None,
                "versus": [],
                "anchor": True,
            }
        )
    rows.sort(key=lambda row: (row["elo"] is None, -(row["elo"] or 0)))
    return {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "anchor": ANCHOR,
        "anchor_rating": ANCHOR_RATING,
        "ladder": {k: ladder.get(k) for k in ("source", "sets", "seed", "imported_at")},
        "submissions": len(results),
        "rows": rows,
    }


STYLE = """
:root{--bg:#fff;--fg:#1b1b1f;--muted:#6b6f76;--line:#e3e5e8;--bar:#3b6ea5;--tint:#f4f6f8;--good:#2e7d32;--bad:#c62828}
@media(prefers-color-scheme:dark){:root{--bg:#121316;--fg:#e8e9ec;--muted:#9aa0a6;--line:#2b2e33;--bar:#7aa7d9;--tint:#1b1d21;--good:#81c784;--bad:#ef9a9a}}
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1040px;margin:0 auto}h1{font-size:24px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 20px}
table{width:100%;border-collapse:collapse}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:13px}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.anchor td{color:var(--muted);background:var(--tint)}tr:has(+ tr.howrow) td{border-bottom:none}tr.howrow td{white-space:normal;padding-top:0}.bar{display:inline-block;height:10px;background:var(--bar);border-radius:3px;vertical-align:middle;margin-right:8px}
.tag{font-size:12px;padding:1px 6px;border:1px solid var(--line);border-radius:10px;color:var(--muted)}.desc{color:var(--muted);font-size:13px;white-space:normal}
.pos{color:var(--good)}.neg{color:var(--bad)}details{margin-top:24px}summary{cursor:pointer;color:var(--muted)}
@media(max-width:700px){th.opt,td.opt{display:none}}
details.how{margin:6px 0 2px}details.how summary{font-size:13px}.scroll{overflow-x:auto;margin:10px 0}
.moves{display:flex;flex-wrap:wrap;gap:14px}.frame{border:1px solid var(--line);border-radius:10px;padding:8px;overflow-x:auto;max-width:100%}
svg text{fill:var(--fg);font:12.5px system-ui,sans-serif}svg .meta{fill:var(--muted);font-size:12px}
.flow rect{stroke-width:1.5}.k-rule{fill:var(--tint);stroke:var(--muted)}.k-net{fill:#3987e522;stroke:#3987e5}
.k-search{fill:#eb683422;stroke:#eb6834}.k-data{fill:#8e7cf022;stroke:#8e7cf0}.k-out{fill:#1baf7a22;stroke:#1baf7a}
.arrow{fill:none;stroke:var(--muted);stroke-width:1.5}
.card{fill:var(--bg);stroke:var(--line);stroke-width:1}.card.legal{stroke:#3987e5;stroke-width:1.6}.card.chosen{stroke:#eda100;stroke-width:3}
.card.dim{opacity:.45}.card.played{stroke:var(--muted)}.cardt{font-size:12px!important}.cardt.red{fill:#e05252}
.bar-net{fill:#3987e5}.bar-search{fill:#eb6834}.neg{opacity:.4}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:4px 0 14px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
"""


SUITS, RANKS = "♣♦♥♠", "23456789TJQKA"
KINDS = {"rule": "k-rule", "net": "k-net", "search": "k-search", "data": "k-data", "out": "k-out"}


def _lines(label: str, width: int = 17) -> list[str]:
    words, lines, cur = label.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return [*lines, cur][:3]


def method_svg(steps: list[list[str]]) -> str:
    """A left-to-right flow of rounded boxes, coloured by kind."""
    if not steps:
        return ""
    bw, gap, h = 150, 28, 64
    width = len(steps) * bw + (len(steps) - 1) * gap
    parts = [f'<svg class="flow" viewBox="0 0 {width} {h + 4}" width="{width}" role="img">']
    for i, (label, kind) in enumerate(steps):
        x = i * (bw + gap)
        parts.append(f'<rect class="{KINDS.get(kind, "k-rule")}" x="{x + 1}" y="2" width="{bw - 2}" height="{h}" rx="10"/>')
        lines = _lines(label)
        y0 = 2 + h / 2 - (len(lines) - 1) * 8
        for j, line in enumerate(lines):
            parts.append(f'<text x="{x + bw / 2}" y="{y0 + j * 16}" text-anchor="middle" dominant-baseline="central">{html.escape(line)}</text>')
        if i < len(steps) - 1:
            ax = x + bw + 3
            parts.append(f'<path class="arrow" d="M{ax} {2 + h / 2}h{gap - 10}m-6 -5l6 5l-6 5"/>')
    parts.append("</svg>")
    return "".join(parts)


def _card(x: float, y: float, c: int, cls: str) -> str:
    red = " red" if c // 13 in (1, 2) else ""
    return (f'<rect class="card {cls}" x="{x}" y="{y}" width="26" height="36" rx="4"/>'
            f'<text class="cardt{red}" x="{x + 13}" y="{y + 12}" text-anchor="middle">{RANKS[c % 13]}</text>'
            f'<text class="cardt{red}" x="{x + 13}" y="{y + 27}" text-anchor="middle">{SUITS[c // 13]}</text>')


def move_svg(frame: dict[str, Any]) -> str:
    """One card decision: the trick so far, the hand (legal cards lit, the played card gold),
    and per-card scores when the bot reports them (net odds, search values)."""
    hand, legal, chosen = frame["hand"], set(frame["legal"]), frame["chosen"]
    scores = frame.get("scores", {})
    width, top = 13 * 28 + 8, 84
    height = top + 40 + len(scores) * 14 + 6
    trump = "NT" if frame["trump"] == 4 else SUITS[frame["trump"]]
    est = "-" if frame["estimate"] is None else frame["estimate"]
    out = [f'<svg class="move" viewBox="0 0 {width} {height}" width="{width}" role="img">',
           f'<text class="meta" x="4" y="14">trump {trump} · won {frame["won"]}/{est}</text>']
    for k, (seat, c) in enumerate(frame["trick"]):
        out.append(f'<text class="meta" x="{4 + k * 34 + 13}" y="{top - 48}" text-anchor="middle">{seat}</text>')
        out.append(_card(4 + k * 34, top - 44, c, "played"))
    if not frame["trick"]:
        out.append(f'<text class="meta" x="4" y="{top - 22}">leads</text>')
    pos = {}
    for k, c in enumerate(sorted(hand, key=lambda c: (c // 13, c % 13))):
        x = 4 + k * 28
        pos[c] = x
        cls = "chosen" if c == chosen else ("legal" if c in legal else "dim")
        out.append(_card(x, top, c, cls))
    for r, (name, vals) in enumerate(sorted(scores.items())):
        y = top + 42 + r * 14
        lo, hi = min(vals.values(), default=0), max(vals.values(), default=1)
        span = (hi - lo) or 1
        for c_str, v in vals.items():
            c = int(c_str)
            if c in pos:  # each row runs from its worst card (short) to its best (full width)
                w = 3 + 21 * (v - lo) / span
                out.append(f'<rect class="bar-{name}" x="{pos[c] + 1}" y="{y}" width="{w:.1f}" height="9" rx="2"/>')
    out.append("</svg>")
    return "".join(out)


def how_it_plays(row: dict[str, Any], samples: dict[str, Any]) -> str:
    frames = samples.get(row["key"], {}).get("frames", [])
    moves = "".join(f'<div class="frame">{move_svg(f)}</div>' for f in frames)
    if not row.get("method") and not moves:
        return ""
    return (f'<details class="how"><summary>how it plays</summary>'
            f'<div class="scroll">{method_svg(row.get("method", []))}</div>'
            f'<div class="moves">{moves}</div></details>')


def render(board: dict[str, Any]) -> str:
    rows = board["rows"]
    rated = [r["elo"] for r in rows if r["elo"] is not None]
    top = max(rated) if rated else 1.0
    anchors = [r["name"] for r in rows if r["anchor"]]
    samples_file = SITE_DIR / "samples.json"
    samples = json.loads(samples_file.read_text()) if samples_file.is_file() else {}
    lines = []
    rank = 0
    for r in rows:
        e = html.escape
        rank += 1  # one global ranking: anchors and submissions alike
        elo = r["elo"]
        bar = f'<span class="bar" style="width:{max(2, int(140 * (elo / top)))}px"></span>' if elo else ""
        elo_text = f"{elo:,.0f}" if elo is not None else "unrated"
        if r["anchor"]:
            name = f'{e(r["name"])} <span class="tag">anchor</span>'
            meta = f'{e(r["author"])}<br><span class="desc">{e(r["description"])}</span>'
            detail = ""
        else:
            name = f'<strong>{e(r["name"])}</strong> <span class="tag">{e(r["kind"])}</span>'
            meta = f'{e(r["author"])}<br><span class="desc">{e(r["description"])}</span>'
            vs = " · ".join(
                f'{e(v["anchor"])} <span class="{"pos" if v["rsa"] >= 0 else "neg"}">{v["rsa"]:+.0f}</span>'
                for v in r["versus"]
            )
            detail = (
                f'<span class="desc">{vs}</span><br><span class="desc">{r["games"]} games, '
                f'seed {r["seed"]}, p95 {r["p95_ms"]:.0f} ms, over budget {r["over_budget"]:.1%}, '
                f'{e(r["played_at"][:10])}</span>'
            )
        lines.append(
            f'<tr class="{"anchor" if r["anchor"] else "sub"}">'
            f'<td class="num">{rank if elo is not None else ""}</td>'
            f"<td>{name}</td><td class=\"opt\">{meta}</td>"
            f'<td class="num">{bar}{elo_text}</td><td class="opt">{detail}</td></tr>'
        )
        how = how_it_plays(r, samples)
        if how:
            cls = "anchor" if r["anchor"] else "sub"
            lines.append(f'<tr class="{cls} howrow"><td></td><td colspan="4">{how}</td></tr>')
    ladder = board["ladder"]
    built = board["built_at"].replace("T", " ")[:16] + " UTC"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estimation Arena</title><style>{STYLE}</style></head>
<body><main>
<h1>Estimation Arena</h1>
<p class="sub">{board["submissions"]} submissions and {len(anchors)} anchors, one Elo scale (Random = {board["anchor_rating"]:.0f}). Built {built}.</p>
<div class="legend"><span><span class="sw" style="background:var(--tint);border:1px solid var(--muted)"></span>rule</span><span><span class="sw" style="background:#3987e5"></span>net</span><span><span class="sw" style="background:#eb6834"></span>search</span><span><span class="sw" style="background:#8e7cf0"></span>data</span><span><span class="sw" style="background:#1baf7a"></span>decision</span><span><span class="sw" style="border:2px solid #3987e5"></span>legal card</span><span><span class="sw" style="border:2px solid #eda100"></span>played</span><span>every bot: same deal, seat 0, vs 3 Heuristic</span></div>
<table>
<thead><tr><th class="num">#</th><th>Bot</th><th class="opt">Author</th><th class="num">Elo</th><th class="opt">Score advantage vs anchors (RSA)</th></tr></thead>
<tbody>
{chr(10).join(lines)}
</tbody></table>
<details><summary>How the numbers are made</summary>
<p class="desc">A submission plays each anchor in duplicate sets: every deal is played four times with the seats rotated, so luck cancels. RSA is the mean score advantage per game over the anchor. Elo is one Bradley-Terry fit over every pairwise set result, including the engine ladder <code>{html.escape(str(ladder.get("source")))}</code> ({ladder.get("sets")} sets), which is what places the anchors. Over budget counts card decisions slower than 0.25 s.</p>
</details>
</main></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.leaderboard")
    parser.add_argument("--out", type=Path, default=SITE_DIR, help="output folder (site/)")
    args = parser.parse_args(argv)
    board = build()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leaderboard.json").write_text(json.dumps(board, indent=1) + "\n")
    (args.out / "index.html").write_text(render(board))
    for r in board["rows"]:
        elo = f"{r['elo']:7.0f}" if r["elo"] is not None else "unrated"
        print(f"{elo}  {r['name']}{'  (anchor)' if r['anchor'] else ''}")
    print(f"wrote {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
