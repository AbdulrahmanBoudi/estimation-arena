"""What every number on the leaderboards means: one explanation, shown on both pages."""

_PLACES = ("King", "Sub-King", "Sub-Kooz", "Kooz")


def _scale(values: tuple[str, str, str, str]) -> str:
    cells = "".join(f'<div class="g-cell g{k}"><span>{_PLACES[k]}</span><b>{v}</b></div>' for k, v in enumerate(values))
    return f'<div class="g-scale" role="img" aria-label="{", ".join(f"{p} {v}" for p, v in zip(_PLACES, values))}">{cells}</div>'


GUIDE = f"""<section class="guide" id="guide" aria-labelledby="guide-title">
<h2 id="guide-title">What the numbers mean</h2>
<p class="g-lead">All three scores come from the same games. They only disagree about what counts as doing well.</p>
<div class="g-metrics">
<article><h3>Weighted rank</h3><p>How good is its finishing order? Each game pays by final place, with the two ends counting most. It's the default ranking.</p>{_scale(("+1", "+0.2", "−0.2", "−1"))}<p class="g-even">An even match averages 0.</p></article>
<article><h3>King only</h3><p>How often does it win the game? First place is all that counts.</p>{_scale(("1", "0", "0", "0"))}<p class="g-even">An even match is 25%.</p></article>
<article><h3>Score delta</h3><p>How much does it win by? This is its average game score minus its opponent's. One huge game can move the average.</p><div class="g-formula"><span>its score</span><b>−</b><span>opponent's score</span></div><p class="g-even">An even match averages 0.</p></article>
</div>
<dl class="g-terms">
<div><dt>Elo</dt><dd>A rating from who won each set, using every match played. A 200-point lead means winning about 76% of sets. Random is fixed at 400 as the reference. Each score has its own Elo, so compare numbers within a column, not across columns.</dd></div>
<div><dt>Set</dt><dd>The same deal played four times with the seats rotated, so lucky cards cancel out. Two copies of each bot sit at the table.</dd></div>
<div><dt>Sets record</dt><dd>Sets won, drawn and lost, in that order. A win under one score isn't always a win under another.</dd></div>
<div><dt>Places</dt><dd>The share of games it finished in each place. Tied seats share the places they occupy.</dd></div>
<div><dt>Anchor</dt><dd>A fixed reference bot that everything else is measured against.</dd></div>
<div><dt>Provisional</dt><dd>Rated only from test matches, so far fewer games stand behind it. Treat a small lead as a tie.</dd></div>
</dl>
</section>"""

# Each page defines --guide-line and --muted; the place colours are the same on both.
STYLE = """.guide{margin-top:36px}.guide h2{margin:0 0 6px}.guide h3{margin:0 0 6px;font-size:17px}
.g-lead{margin:0 0 16px;max-width:70ch}
.g-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:24px 32px;margin-bottom:24px}
.g-metrics p{margin:0}.g-metrics article>p:first-of-type{min-height:6em}
.g-scale{display:grid;grid-template-columns:repeat(4,1fr);gap:2px;margin:10px 0 8px}
.g-cell{padding:6px 4px;border-radius:3px;text-align:center;color:#1c2147;font-size:12px;line-height:1.3}.g-cell b{display:block;font-size:16px}
.g0{background:#f2c14e}.g1{background:#f8e3a8}.g2{background:#f3a6a8}.g3{background:#e5484d;color:#fff}
.g-formula{display:flex;gap:10px;align-items:center;margin:10px 0 8px;min-height:54px}.g-formula span{flex:1;padding:8px 4px;text-align:center;border:1px solid var(--guide-line);border-radius:3px;font-size:13px}
.g-even{color:var(--muted);font-size:14px}
.g-terms{margin:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px 32px}.g-terms dt{font-weight:600}.g-terms dd{margin:2px 0 0;color:var(--muted);font-size:14px}
@media(max-width:680px){.g-metrics article>p:first-of-type{min-height:0}}
"""
