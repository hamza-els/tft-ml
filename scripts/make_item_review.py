"""
Make a review page for the item templates under units' health bars.

Usage:
    python scripts/make_item_review.py            # -> data/raw/review/item_review.html

Shows every pair of templates in assets/templates/unit_items/ that look alike
(src/vision/unit_bar.item_similarity >= MIN_SIM), most alike first, with
Same / Different buttons, plus a name box for each unnamed ID. "Copy result"
gives a JSON text to paste back; scripts/apply_item_review.py applies it
(renames the files so each item has one ID).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import base64
import html
import json

import cv2
import numpy as np

from src.vision.items import _ROOT
from src.utils.config import vision_cfg
from src.vision.unit_bar import item_hue, item_vec

FOLDER = _ROOT / vision_cfg["unit_bar"]["item_templates"]
OUT = _ROOT / "data" / "raw" / "review" / "item_review.html"
MIN_SIM = 0.40
MAX_PAIRS = 150


def data_uri(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def main():
    files = sorted(FOLDER.glob("*.png"))
    imgs = [cv2.imread(str(f)) for f in files]
    stems = [f.stem for f in files]
    ids = [s.split("__")[0] for s in stems]
    shape = np.stack([item_vec(i) for i in imgs])
    hue = np.stack([item_hue(i) for i in imgs])
    sim = (shape @ shape.T) * (hue @ hue.T)
    pairs = sorted(((float(sim[i, j]), i, j) for i in range(len(files)) for j in range(i + 1, len(files))
                    if ids[i] != ids[j] and sim[i, j] >= MIN_SIM), reverse=True)[:MAX_PAIRS]
    uris = [data_uri(i) for i in imgs]

    def icon(i):
        label = html.escape(ids[i])
        cls = "unnamed" if ids[i].startswith("uitem_") else "named"
        return (f'<figure><img src="{uris[i]}" alt="{label}"><figcaption class="{cls}">{label}'
                f'</figcaption></figure>')

    rows = []
    for n, (s, i, j) in enumerate(pairs, 1):
        rows.append(
            f'<div class="pair" data-a="{html.escape(ids[i])}" data-b="{html.escape(ids[j])}">'
            f'<span class="num">#{n}</span>{icon(i)}{icon(j)}'
            f'<span class="score">{s:.2f}</span>'
            f'<div class="btns"><button class="same">Same</button><button class="diff">Different</button></div>'
            f'</div>')
    unnamed = sorted({i for i, x in enumerate(ids) if x.startswith("uitem_")}, key=lambda i: ids[i])
    name_rows = [f'<div class="nameit">{icon(i)}<input data-id="{html.escape(ids[i])}" '
                 f'placeholder="item name (optional)"></div>' for i in unnamed]

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Item template review</title>
<style>
:root {{ --bg:#1b1c21; --card:#26272e; --fg:#e8e8ea; --dim:#9a9ca6; --same:#3c8d5a; --diff:#8d3c3c; --accent:#f2c14e; }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:14px system-ui,sans-serif; }}
h1 {{ font-size:20px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:28px 0 8px; }}
p {{ color:var(--dim); max-width:760px; margin:4px 0 12px; }}
.grid {{ display:flex; flex-wrap:wrap; gap:10px; }}
.pair, .nameit {{ background:var(--card); border-radius:8px; padding:8px 10px; display:flex; align-items:center; gap:10px; border:2px solid transparent; }}
.pair.is-same {{ border-color:var(--same); }} .pair.is-diff {{ border-color:var(--diff); opacity:.55; }}
figure {{ margin:0; text-align:center; width:88px; }}
img {{ width:75px; height:75px; image-rendering:pixelated; border-radius:4px; }}
figcaption {{ font-size:11px; color:var(--dim); word-break:break-word; }} figcaption.named {{ color:var(--fg); }}
.num {{ color:var(--dim); width:34px; }} .score {{ color:var(--accent); font-variant-numeric:tabular-nums; }}
.btns {{ display:flex; flex-direction:column; gap:4px; }}
button {{ background:#33353f; color:var(--fg); border:0; border-radius:5px; padding:5px 10px; cursor:pointer; }}
.pair.is-same .same {{ background:var(--same); }} .pair.is-diff .diff {{ background:var(--diff); }}
input {{ background:#33353f; color:var(--fg); border:0; border-radius:5px; padding:6px 8px; width:170px; }}
#out {{ width:100%; max-width:760px; height:120px; background:#111216; color:var(--fg); border:1px solid #333; border-radius:6px; }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; z-index:1; display:flex; gap:10px; align-items:center; }}
</style></head><body>
<h1>Item template review</h1>
<p>Each pair below is two item templates that look alike, most alike first (score 0-1). Mark <b>Same</b> when both
are pictures of the same item, <b>Different</b> when not; unanswered pairs are left as they are. Radiant and normal
versions are different items. Then name any unnamed IDs you recognise (optional), press <b>Copy result</b>, and paste
it back in the chat.</p>
<div class="bar"><button id="copy">Copy result</button><span id="count" style="color:var(--dim)"></span></div>
<h2>Look-alike pairs ({len(pairs)})</h2>
<div class="grid">{''.join(rows)}</div>
<h2>Unnamed IDs ({len(unnamed)})</h2>
<div class="grid">{''.join(name_rows)}</div>
<h2>Result</h2>
<textarea id="out" readonly></textarea>
<script>
const pairs = [...document.querySelectorAll('.pair')];
function result() {{
  const same = [], diff = [], names = {{}};
  for (const p of pairs) {{
    if (p.classList.contains('is-same')) same.push([p.dataset.a, p.dataset.b]);
    if (p.classList.contains('is-diff')) diff.push([p.dataset.a, p.dataset.b]);
  }}
  for (const i of document.querySelectorAll('input[data-id]')) if (i.value.trim()) names[i.dataset.id] = i.value.trim();
  return {{same, different: diff, names}};
}}
function update() {{
  const r = result();
  document.getElementById('out').value = JSON.stringify(r);
  document.getElementById('count').textContent = `${{r.same.length}} same, ${{r.different.length}} different, ${{Object.keys(r.names).length}} named`;
}}
for (const p of pairs) {{
  p.querySelector('.same').onclick = () => {{ p.classList.toggle('is-same'); p.classList.remove('is-diff'); update(); }};
  p.querySelector('.diff').onclick = () => {{ p.classList.toggle('is-diff'); p.classList.remove('is-same'); update(); }};
}}
document.querySelectorAll('input[data-id]').forEach(i => i.oninput = update);
document.getElementById('copy').onclick = async () => {{
  update(); const t = document.getElementById('out');
  try {{ await navigator.clipboard.writeText(t.value); }} catch (e) {{ t.select(); document.execCommand('copy'); }}
  document.getElementById('copy').textContent = 'Copied';
  setTimeout(() => document.getElementById('copy').textContent = 'Copy result', 1500);
}};
update();
</script></body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"{len(files)} templates, {len(set(ids))} IDs, {len(pairs)} look-alike pairs, "
          f"{len(unnamed)} unnamed -> {OUT}")


if __name__ == "__main__":
    main()
