"""
Items on units over time (Hamza, 2026-11-24): what each unit holds, and the
item events the gameplay model needs (equip, combine, remove/reforge, sell).

An item can't come off a unit on its own. It leaves only by
  - a Magnetic Remover or a Reforger (the column's count of them is not
    read: a stack shows as one icon), or by selling the unit: either way the
    unit's items go back to the item column, so the column gains items on
    the same frames (a reforger returns other items than the unit had);
  - combining: a component on the unit and a second one dropped on it become
    a completed item (recipes from the set data).
So when a unit's reading drops an item and the column gains nothing around
that frame, the item was hidden or misread: it is kept on the unit
("items_kept"). Items are compared by base name (Radiant = base + flag).
Exempt: Thief's Gloves holders (their items change every round).

Units are followed from read frame to read frame: same spot and champion,
else the same champion (and star) somewhere else (moved). A unit that
stars up starts a new track (combining moves the copies' items together).
"""

from collections import Counter

ITEM_COLUMN_WINDOW = 2      # read frames either side searched for the column gaining items
KEEP_MISSING_FRAMES = 3     # a kept item missing this many read frames in a row is gone after all
FREE_ITEM_HOLDERS = {"Thief's Gloves"}


def _pos(u: dict) -> tuple:
    return ("h", *u["hex"]) if "hex" in u else ("s", u["slot"])


def _column(rows: list[dict], read: list[int]) -> list[Counter]:
    """Item column per read frame, steadied: each item counted as in the middle of 3 readings."""
    raw = [Counter(x for x in rows[i].get("bench_items") or [] if x) for i in read]
    out = []
    for k in range(len(raw)):
        near = raw[max(0, k - 1):k + 2]
        names = set().union(*near)
        out.append(Counter({n: sorted(c[n] for c in near)[len(near) // 2] for n in names}))
        out[-1] = +out[-1]
    return out


def track_items(out: list[dict], rows: list[dict], read: list[int], game, stats: dict) -> None:
    """
    In place: units' items in `out` (clean_units' per-row board/bench) kept or
    accepted as above, and out[i]["item_events"] on read rows:
    {"kind": equip | combine | removed | reforged | sold | lost, "name", "at", "items"};
    "lost": missing KEEP_MISSING_FRAMES read frames in a row with nothing back in the column
    (uncertain: by eye often a neighbour's items or a misread icon, not a removal).
    """
    base = (lambda n: game.item_base(n)[0]) if game is not None else (lambda n: n)
    recipe = (lambda n: (game.items.get(n) or {}).get("recipe") or []) if game is not None else (lambda n: [])
    col = _column(rows, read)

    def column_gain(k: int) -> Counter:
        before = col[max(0, k - ITEM_COLUMN_WINDOW)]
        gain = Counter()
        for j in range(k, min(len(col), k + ITEM_COLUMN_WINDOW + 1)):
            gain |= Counter({base(n): c for n, c in (col[j] - before).items()})
        return gain

    tracks: dict[int, dict] = {}          # id -> {"name", "star", "pos", "items"}
    prev: list[tuple[int, dict]] = []     # (track id, unit) on the previous read frame
    next_id = 0
    for k, i in enumerate(read):
        events = []
        units = out[i]["board"] + out[i]["bench"]
        assigned: dict[int, int] = {}     # index in units -> track id
        free_ids = {tid for tid, _ in prev}
        for n, u in enumerate(units):     # same spot, same champion and star
            tid = next((t for t, v in prev if t in free_ids and _pos(v) == _pos(u)
                        and v["name"] == u["name"] and v.get("star") == u.get("star")), None)
            if tid is not None:
                assigned[n] = tid
                free_ids.discard(tid)
        for n, u in enumerate(units):     # moved: the one copy of that champion and star that moved
            if n in assigned:
                continue
            same = lambda v: v["name"] == u["name"] and v.get("star") == u.get("star")
            olds = [t for t, v in prev if t in free_ids and same(v)]
            news = [m for m, v in enumerate(units) if m not in assigned and same(v)]
            if len(olds) == 1 and len(news) == 1:   # two copies moving: can't tell which is which
                assigned[n] = olds[0]
                free_ids.discard(olds[0])
        gain = column_gain(k)
        for tid in free_ids:              # gone: sold (its items back to the column) or moved off view
            t = tracks[tid]
            held = Counter(base(x) for x in t["items"])
            if held and not (held - gain):
                events.append({"kind": "sold", "name": t["name"], "at": list(t["pos"]), "items": t["items"]})
                stats["item_sold"] += 1
        cur = []
        for n, u in enumerate(units):
            items = [x for x in u.get("items") or [] if x]
            tid = assigned.get(n)
            if tid is None:
                tid, next_id = next_id, next_id + 1
                tracks[tid] = {"name": u["name"], "star": u.get("star"), "pos": _pos(u), "items": items}
                cur.append((tid, u))
                continue
            t = tracks[tid]
            before = t["items"]
            if FREE_ITEM_HOLDERS & {base(x) for x in before + items}:
                t.update(pos=_pos(u), items=items)
                cur.append((tid, u))
                continue
            lost = Counter(base(x) for x in before) - Counter(base(x) for x in items)
            got = Counter(base(x) for x in items) - Counter(base(x) for x in before)
            for x in list(got):           # a component on the unit + one dropped on it = completed item
                parts = recipe(x)
                if parts and any(lost[p] for p in parts):
                    for p in parts:
                        if lost[p]:
                            lost[p] -= 1
                    lost = +lost
                    events.append({"kind": "combine", "name": u["name"], "at": list(_pos(u)), "items": [x]})
                    stats["item_combine"] += 1
                    got[x] -= 1
            got = +got
            if lost:
                if not (lost - gain):
                    kind = "removed"
                elif sum(gain.values()) >= sum(lost.values()):
                    kind = "reforged"
                else:
                    kind = None
                if kind:
                    events.append({"kind": kind, "name": u["name"], "at": list(_pos(u)),
                                   "items": sorted(lost.elements())})
                    stats[f"item_{kind}"] += 1
                else:                     # nothing came back to the column: hidden or misread, kept
                    missing = t.setdefault("missing", Counter())
                    for x in lost:
                        missing[x] += 1
                    gone = {x for x in lost if missing[x] >= KEEP_MISSING_FRAMES}
                    if gone:              # missing too long: it did come off (column reading missed it)
                        # "lost", not "removed": checked by eye, these are often a neighbour's
                        # items or a misread icon rather than a real removal
                        events.append({"kind": "lost", "name": u["name"], "at": list(_pos(u)),
                                       "items": sorted(gone)})
                        stats["item_lost"] += 1
                    keep = [x for x in before if base(x) in lost and base(x) not in gone]
                    room = 3 - len(items)
                    if keep and room > 0:
                        u["items"] = items + keep[:room]
                        u["items_kept"] = True
                        stats["items_kept"] += min(room, len(keep))
                        items = u["items"]
            for x in [x for x in t.get("missing", {}) if x not in lost]:
                del t["missing"][x]
            for x in got.elements():
                events.append({"kind": "equip", "name": u["name"], "at": list(_pos(u)), "items": [x]})
                stats["item_equip"] += 1
            t.update(pos=_pos(u), items=items)
            cur.append((tid, u))
        prev = cur
        if events:
            out[i]["item_events"] = events
