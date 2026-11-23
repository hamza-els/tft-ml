"""
Review page for the names of the item-column templates (assets/templates/items_seen/).

Most of them were named automatically (matched to official art when cut), so nobody has
confirmed them by eye. The page shows each named template next to the official art of its
name, how often it is read, and one item column where it was read. Mark only the wrong ones
(pick the right name, "not an item" or "can't tell"), then "Save result" downloads
item_check.json; scripts/apply_item_check.py applies it.

Usage:
    python scripts/make_item_check.py          # -> data/raw/review/item_check.html
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import base64
import html
import json
from collections import Counter

import cv2
import numpy as np

from src.state.game_data import load_game_data
from src.utils import paths
from src.utils.config import vision_cfg
from src.vision.items import ItemColumnReader

SEEN = paths.REPO / vision_cfg["item_column"]["seen_dir"]
OFFICIAL = paths.REPO / vision_cfg["template_matching"]["item_dir"]
OUT = paths.REPO / "data" / "raw" / "review" / "item_check.html"
LOG = paths.LOCAL / "item_template_rebuild" / "name_log.json"
N_SUGGEST = 6


def base(name: str) -> str:
    return name.split("~")[0]


def uri(img: np.ndarray, ext: str = ".png") -> str:
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 85] if ext == ".jpg" else [])
    mime = "jpeg" if ext == ".jpg" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buf.tobytes()).decode()


def big(img: np.ndarray, px: int) -> np.ndarray:
    return cv2.resize(img, (px, px), interpolation=cv2.INTER_NEAREST)


def readings() -> tuple[Counter, dict[str, tuple[str, str, int]]]:
    """Readings per item name in the local state files, and one frame (in a VOD with full
    frames) where each name was read."""
    counts: Counter = Counter()
    where: dict[str, tuple[str, str, int]] = {}
    for f in sorted(paths.STATES.glob("*/game_??.jsonl")):
        vod, game = f.parent.name, f.stem
        has_frames = (paths.FRAMES / vod / game).is_dir()
        for line in f.read_text(encoding="utf-8").splitlines():
            if '"bench_items": [' not in line:
                continue
            r = json.loads(line)
            for n in r.get("bench_items") or []:
                if n:
                    counts[n] += 1
                    if has_frames and n not in where:
                        where[n] = (vod, game, r["frame"])
    return counts, where


def column_crop(reader: ItemColumnReader, vod: str, game: str, frame: int) -> np.ndarray | None:
    img = cv2.imread(str(paths.FRAMES / vod / game / f"frame_{frame:06d}.jpg"))
    if img is None:
        return None
    x0, y0 = max(0, reader.x - 20), max(0, reader.y - 20)
    y1 = int(reader.y + reader.slots * reader.pitch + reader.size + 20)
    crop = img[y0:y1, x0:reader.x + reader.size + 160]
    return cv2.resize(crop, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)


def main():
    game = load_game_data()
    reader = ItemColumnReader()
    all_names = sorted(set(game.items) | set(game.emblems))
    try:
        carried = set(json.loads(LOG.read_text(encoding="utf-8")).values())
    except (OSError, ValueError):
        carried = set()
    counts, where = readings()

    off = reader.official
    off_m = np.stack(off._rows)
    seen_named = [(n, r) for n, r in zip(reader.seen.names, reader.seen._rows) if not n.startswith("item_")]

    cards = []
    for p in sorted(SEEN.glob("*.png")):
        if p.stem.startswith("item_"):
            continue
        icon = cv2.imread(str(p))
        name = base(p.stem)
        vec = off.prepare(icon)
        # Alternatives: official art and other named footage templates, best score per name.
        best: dict[str, float] = {}
        for n, s in zip(off.names, off_m @ vec):
            if n in game.items or n in game.emblems:
                best[n] = max(best.get(n, -1.0), float(s))
        for n, r in seen_named:
            if n != p.stem and base(n) in all_names:
                best[base(n)] = max(best.get(base(n), -1.0), float(r @ vec))
        own = best.pop(name, None)
        alts = sorted(best.items(), key=lambda kv: -kv[1])[:N_SUGGEST]
        art = cv2.imread(str(OFFICIAL / f"{name}.png"))
        ctx = column_crop(reader, *where[name]) if name in where else None
        cards.append({
            "t": p.stem, "name": name, "reads": counts.get(name, 0),
            # Cut templates get an official name only at >= official_threshold; any other name
            # was given by hand (names_review.json, applied) or carried over from the old set.
            "src": ("carried over" if p.stem in carried else "official-art match"
                    if own is not None and own >= reader.official_threshold else "your label"),
            "own": own,
            "img": uri(big(icon, 132)),
            "art": uri(big(art, 132)) if art is not None else None,
            "ctx": uri(ctx, ".jpg") if ctx is not None else None,
            "alts": [{"n": n, "s": round(s, 2),
                      "img": (uri(big(a, 44)) if (a := cv2.imread(str(OFFICIAL / f"{n}.png"))) is not None else None)}
                     for n, s in alts],
        })
    # Unverified first (official-art matches), then most read.
    cards.sort(key=lambda c: (c["src"] != "official-art match", -c["reads"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(cards))
                   .replace("__NAMES__", json.dumps(all_names)), encoding="utf-8")
    print(f"{len(cards)} named templates -> {OUT}")


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Item template check</title>
<style>
:root{--bg:#1e1f24;--card:#2a2b31;--fg:#e6e6e6;--dim:#9a9aa5;--acc:#7fd1ff;--bad:#ff7a7a;--ok:#7fdc9a}
body{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:0;padding:16px}
header{position:sticky;top:0;background:var(--bg);padding:8px 0 12px;z-index:2;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0}
button{background:#3a3c44;color:var(--fg);border:1px solid #555;border-radius:6px;padding:6px 10px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)}
button.save{background:#22506a;border-color:var(--acc)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:12px}
.card{background:var(--card);border-radius:10px;padding:12px;border:2px solid transparent}
.card.wrong{border-color:var(--bad)}
.row{display:flex;gap:10px;align-items:flex-start}
.pics img{display:block;border-radius:4px;image-rendering:pixelated}
.pics small{color:var(--dim);display:block;text-align:center;margin-top:2px}
.noart{width:132px;height:132px;display:flex;align-items:center;justify-content:center;color:var(--dim);border:1px dashed #555;border-radius:4px;text-align:center}
.ctx img{max-height:300px;border-radius:4px}
.name{font-size:16px;font-weight:600;margin:8px 0 2px}
.meta{color:var(--dim);font-size:12px}
.fix{display:none;margin-top:10px}
.card.wrong .fix{display:block}
.alts{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}
.alt{display:flex;align-items:center;gap:6px;padding:3px 8px 3px 3px}
.alt img{width:36px;height:36px;border-radius:3px}
.alt.sel,.sp.sel{border-color:var(--bad);background:#4a2a2a}
input{background:#15161a;color:var(--fg);border:1px solid #555;border-radius:6px;padding:6px;width:260px;font:inherit}
.chosen{color:var(--bad);margin-top:6px;font-weight:600}
</style></head><body>
<header><h1>Item template check</h1>
<span id="stat" class="meta"></span>
<button class="save" onclick="save()">Save result</button>
<span class="meta">Click "Wrong" on a card only if the name is wrong. Radiant vs normal counts.</span></header>
<div class="grid" id="grid"></div>
<datalist id="names"></datalist>
<script>
const CARDS=__DATA__, NAMES=__NAMES__;
let ans={};
try{ans=JSON.parse(localStorage.getItem('item_check')||'{}')}catch(e){}
const keep=()=>{try{localStorage.setItem('item_check',JSON.stringify(ans))}catch(e){}};
document.getElementById('names').innerHTML=NAMES.map(n=>`<option value="${n.replace(/"/g,'&quot;')}">`).join('');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function render(){
  const g=document.getElementById('grid');
  g.innerHTML=CARDS.map((c,i)=>{
    const a=ans[c.t], wrong=a!==undefined;
    const label=a==='_none'?'not an item':a==='_unknown'?"an item, can't tell":a==null?'':a;
    return `<div class="card ${wrong?'wrong':''}">
     <div class="row pics">
       <div><img src="${c.img}" width="132" height="132"><small>footage</small></div>
       <div>${c.art?`<img src="${c.art}" width="132" height="132">`:`<div class="noart">no official art</div>`}<small>official: name</small></div>
       ${c.ctx?`<div class="ctx"><img src="${c.ctx}"><small>a column where it was read</small></div>`:''}
     </div>
     <div class="name">${esc(c.name)}</div>
     <div class="meta">${esc(c.t)} · ${c.src} · name read ${c.reads.toLocaleString()} times${c.own!=null?` · art match ${c.own.toFixed(2)}`:''}</div>
     <div style="margin-top:8px"><button onclick="flip(${i})">${wrong?'Undo (name is right)':'Wrong'}</button></div>
     <div class="fix">
       <div class="meta">Right name: pick one, or type it</div>
       <div class="alts">${c.alts.map(x=>`<button class="alt ${a===x.n?'sel':''}" onclick="pick(${i},this.dataset.n)" data-n="${esc(x.n)}">${x.img?`<img src="${x.img}">`:''}${esc(x.n)} <span class="meta">${x.s}</span></button>`).join('')}</div>
       <input list="names" placeholder="type an item name" value="${a&&!a.startsWith('_')&&!c.alts.some(x=>x.n===a)?esc(a):''}" onchange="typed(${i},this.value)">
       <div class="alts"><button class="sp ${a==='_none'?'sel':''}" onclick="pick(${i},'_none')">not an item</button>
       <button class="sp ${a==='_unknown'?'sel':''}" onclick="pick(${i},'_unknown')">an item, can't tell which</button></div>
       <div class="chosen">${label?'→ '+esc(label):'(choose the right name)'}</div>
     </div></div>`}).join('');
  const n=Object.keys(ans).length, open=Object.values(ans).filter(v=>v===null).length;
  document.getElementById('stat').textContent=`${CARDS.length} templates · ${n} marked wrong${open?` (${open} still need a name)`:''}`;
}
function flip(i){const t=CARDS[i].t; if(t in ans) delete ans[t]; else ans[t]=null; keep(); render()}
function pick(i,n){ans[CARDS[i].t]=n; keep(); render()}
function typed(i,v){v=v.trim(); if(!v) return; if(!NAMES.includes(v)){alert('Not an item name: '+v); return} pick(i,v)}
function save(){
  const open=Object.entries(ans).filter(([k,v])=>v===null).map(([k])=>k);
  if(open.length && !confirm(open.length+' marked wrong without a name (saved as "can\'t tell"). Save anyway?')) return;
  const out={}; for(const [k,v] of Object.entries(ans)) out[k]=v===null?'_unknown':v;
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));
  a.download='item_check.json'; a.click();
}
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
