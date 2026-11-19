"""
Review page for the augment templates cut from footage (assets/templates/augments/):
merge duplicates and name them from the official augment icons.

Usage:
    python scripts/make_augment_review.py        # -> data/raw/review/augment_review.html

Two sections:
  1. Look-alike template pairs (Same / Different), most alike first.
  2. Every template still named aug_XXX next to its three closest official
     icons (assets/templates/augments_official/, scripts/fetch_augment_icons.py):
     click the right one, type a name, or leave it.
"Copy result" gives JSON to paste back; apply it with
    python scripts/apply_item_review.py RESULT.json --kind augments

Similarity: shape x colour (src/vision/unit_bar.item_similarity's measure) on
both icons resized to 32 px.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import base64
import html
import json

import cv2
import numpy as np

from src.vision.unit_bar import item_hue, item_vec

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "assets" / "templates" / "augments"
OFFICIAL = ROOT / "assets" / "templates" / "augments_official"
OUT = ROOT / "data" / "raw" / "review" / "augment_review.html"
SIZE = 32
MIN_PAIR = 0.45
MAX_PAIRS = 120


def norm(img):
    """Square centre of the icon at SIZE px."""
    h, w = img.shape[:2]
    s = min(h, w)
    sq = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def feats(img):
    n = norm(img)
    return item_vec(n), item_hue(n)


def uri(img, size=75):
    ok, buf = cv2.imencode(".png", cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def main():
    files = sorted(FOLDER.glob("*.png"))
    imgs = [cv2.imread(str(f)) for f in files]
    ids = [f.stem.split("__")[0] for f in files]
    F = [feats(i) for i in imgs]
    off = {p.stem: (cv2.imread(str(p)),) for p in sorted(OFFICIAL.glob("*.png"))}
    off_f = {k: feats(v[0]) for k, v in off.items()}

    def sim(a, b):
        return float(a[0] @ b[0]) * float(a[1] @ b[1])

    pairs = sorted(((sim(F[i], F[j]), i, j) for i in range(len(files)) for j in range(i + 1, len(files))
                    if ids[i] != ids[j] and sim(F[i], F[j]) >= MIN_PAIR), reverse=True)[:MAX_PAIRS]

    def fig(i):
        return f'<figure><img src="{uri(imgs[i])}"><figcaption>{html.escape(ids[i])}</figcaption></figure>'

    pair_rows = [f'<div class="pair" data-a="{html.escape(ids[i])}" data-b="{html.escape(ids[j])}">'
                 f'<span class="num">#{n}</span>{fig(i)}{fig(j)}<span class="score">{s:.2f}</span>'
                 f'<div class="btns"><button class="same">Same</button><button class="diff">Different</button></div></div>'
                 for n, (s, i, j) in enumerate(pairs, 1)]
    name_rows, seen = [], set()
    for i, tid in enumerate(ids):
        if not tid.startswith("aug_") or tid in seen:
            continue
        seen.add(tid)
        top = sorted(((sim(F[i], v), k) for k, v in off_f.items()), reverse=True)[:3]
        opts = "".join(
            f'<button class="opt" data-name="{html.escape(k.split("__")[0])}"><img src="{uri(off[k][0])}">'
            f'<span>{html.escape(k.split("__")[0])}<br><small>{k.split("__")[-1]} &middot; {s:.2f}</small></span></button>'
            for s, k in top)
        name_rows.append(f'<div class="row" data-id="{html.escape(tid)}">{fig(i)}{opts}'
                         f'<input placeholder="or type the name"><button class="junk">not an augment</button></div>')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Augment template review</title>
<style>
:root {{ --bg:#1b1c21; --card:#26272e; --fg:#e8e8ea; --dim:#9a9ca6; --same:#3c8d5a; --diff:#8d3c3c; --accent:#f2c14e; }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:14px system-ui,sans-serif; }}
p {{ color:var(--dim); max-width:780px; }} h2 {{ font-size:16px; margin:24px 0 8px; }}
.grid {{ display:flex; flex-wrap:wrap; gap:10px; }}
.pair, .row {{ background:var(--card); border-radius:8px; padding:8px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; border:2px solid transparent; }}
.row {{ margin-bottom:8px; }}
.pair.is-same {{ border-color:var(--same); }} .pair.is-diff {{ border-color:var(--diff); opacity:.55; }}
.row.is-junk {{ border-color:var(--diff); opacity:.55; }}
figure {{ margin:0; text-align:center; width:88px; }} figcaption {{ font-size:11px; color:var(--dim); }}
img {{ width:75px; height:75px; image-rendering:pixelated; border-radius:4px; }}
.num {{ color:var(--dim); }} .score {{ color:var(--accent); }}
button {{ background:#33353f; color:var(--fg); border:0; border-radius:5px; padding:5px 10px; cursor:pointer; }}
.pair.is-same .same {{ background:var(--same); }} .pair.is-diff .diff {{ background:var(--diff); }}
.opt {{ display:flex; gap:8px; align-items:center; border:2px solid transparent; width:240px; text-align:left; }}
.opt.picked {{ border-color:var(--same); background:#2c3a31; }}
small {{ color:var(--dim); }}
input {{ background:#33353f; color:var(--fg); border:0; border-radius:5px; padding:6px 8px; width:170px; }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; z-index:1; }}
#out {{ width:100%; max-width:780px; height:90px; background:#111216; color:var(--fg); }}
</style></head><body>
<h1 style="font-size:20px">Augment template review</h1>
<p>Augment icons cut from the footage. 1: pairs that look alike - <b>Same</b> if both are the same augment. 2: each
unnamed template next to the three closest official icons (with their tier) - click the right one, type a name if
none fit, mark <b>not an augment</b> for junk, or leave it. Then <b>Copy result</b> and paste it in the chat.</p>
<div class="bar"><button id="copy">Copy result</button> <span id="count"></span></div>
<h2>1. Look-alike pairs ({len(pair_rows)})</h2><div class="grid">{''.join(pair_rows)}</div>
<h2>2. Name the templates ({len(name_rows)})</h2>{''.join(name_rows)}
<textarea id="out" readonly></textarea>
<script>
function result() {{
  const same=[], different=[], names={{}}, junk=[];
  for (const p of document.querySelectorAll('.pair')) {{
    if (p.classList.contains('is-same')) same.push([p.dataset.a, p.dataset.b]);
    if (p.classList.contains('is-diff')) different.push([p.dataset.a, p.dataset.b]); }}
  for (const r of document.querySelectorAll('.row')) {{
    if (r.classList.contains('is-junk')) {{ junk.push(r.dataset.id); continue; }}
    const t = r.querySelector('input').value.trim(), p = r.querySelector('.opt.picked');
    if (t) names[r.dataset.id] = t; else if (p) names[r.dataset.id] = p.dataset.name; }}
  return {{same, different, names, junk}};
}}
function update() {{ const r = result(); document.getElementById('out').value = JSON.stringify(r);
  document.getElementById('count').textContent = `${{r.same.length}} same, ${{Object.keys(r.names).length}} named, ${{r.junk.length}} junk`; }}
for (const p of document.querySelectorAll('.pair')) {{
  p.querySelector('.same').onclick = () => {{ p.classList.toggle('is-same'); p.classList.remove('is-diff'); update(); }};
  p.querySelector('.diff').onclick = () => {{ p.classList.toggle('is-diff'); p.classList.remove('is-same'); update(); }}; }}
for (const r of document.querySelectorAll('.row')) {{
  for (const b of r.querySelectorAll('.opt')) b.onclick = () => {{ const was = b.classList.contains('picked');
    r.querySelectorAll('.opt').forEach(x => x.classList.remove('picked')); if (!was) b.classList.add('picked'); update(); }};
  r.querySelector('.junk').onclick = () => {{ r.classList.toggle('is-junk'); update(); }};
  r.querySelector('input').oninput = update; }}
document.getElementById('copy').onclick = async () => {{ update(); const t = document.getElementById('out');
  try {{ await navigator.clipboard.writeText(t.value); }} catch (e) {{ t.select(); document.execCommand('copy'); }}
  document.getElementById('copy').textContent = 'Copied'; }};
update();
</script></body></html>""", encoding="utf-8")
    print(f"{len(files)} templates; {len(pair_rows)} look-alike pairs; {len(name_rows)} to name -> {OUT}")


if __name__ == "__main__":
    main()
