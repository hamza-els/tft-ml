"""
Game events worked out from the readings, for cleaning and for the gameplay
model's actions (Hamza, 2026-11-24).

purchases(rows): champions bought from the shop. A card is bought when its
slot turns empty while the other cards stay as they were (a reroll changes
them all), stays empty on the next shop reading, and, where gold is read
before and after, gold dropped by exactly the cost of the cards bought
there. The shop showing no card at all is not read as purchases: that is
the sell panel (a unit being dragged) or a hidden shop.
"""

from collections import Counter

GOLD_LOOK = 4          # rows either side searched for a gold reading


def _shop(row: dict) -> list | None:
    """The shop's 5 slots, or None where it can't be compared (hidden, or the sell panel)."""
    slots = row.get("shop_slots") or []
    if not row.get("shop_visible") or len(slots) != 5 or not any(slots):
        return None
    return list(slots)


def _gold_near(rows: list[dict], i: int, step: int) -> float | None:
    for j in range(i, max(-1, i - GOLD_LOOK) if step < 0 else min(len(rows), i + GOLD_LOOK), step):
        if rows[j].get("gold") is not None and rows[j].get("shop_visible"):
            return rows[j]["gold"]
    return None


def purchases(rows: list[dict], game) -> list[dict]:
    """
    Champions bought, in frame order: {"row": index of the first row showing
    the empty slot, "frame", "name", "slot", "confirmed": gold dropped by the
    cards' cost (False: gold not read around it)}. rows: one game's rows in
    frame order (all of them, not only planning: units can be bought in combat).
    """
    shops = [(i, s) for i, s in ((i, _shop(r)) for i, r in enumerate(rows)) if s is not None]
    out = []
    for n in range(1, len(shops) - 1):
        (i0, a), (i1, b), (_, c) = shops[n - 1], shops[n], shops[n + 1]
        gone = [k for k in range(5) if a[k] and b[k] is None and c[k] is None]
        if not gone:
            continue
        if any(a[k] != b[k] for k in range(5) if k not in gone and a[k] and b[k]):
            continue                                       # other cards changed: a reroll
        names = [a[k] for k in gone]
        try:
            cost = sum(game.cost(nm) for nm in names)
        except (KeyError, ValueError):
            continue                                       # not a champion name (OCR misread)
        before, after = _gold_near(rows, i0, -1), _gold_near(rows, i1, 1)
        confirmed = before is not None and after is not None and before - after == cost
        if before is not None and after is not None and not confirmed and before - after < cost:
            continue                                       # gold didn't drop enough: not bought
        for k, nm in zip(gone, names):
            out.append({"row": i1, "frame": rows[i1]["frame"], "name": nm, "slot": k, "confirmed": confirmed})
    return out


def bought_by_row(rows: list[dict], game) -> list[list[str]]:
    """Per row, the champions bought on it (purchases())."""
    out = [[] for _ in rows]
    for p in purchases(rows, game):
        out[p["row"]].append(p["name"])
    return out
