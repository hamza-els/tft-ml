"""
Clean one game's sequence of per-frame states (rows from scripts/extract_states.py).

Per-frame OCR is occasionally wrong or missing (bottom bar hidden, a tooltip
over a number). A single frame can't tell, but the sequence can, because the
game constrains how values move:

  stage/round  never go backwards
  level        never goes down
  hp           usually falls but can rise (some augments heal), so no
               direction is assumed; a reading is dropped only when it is a
               short blip between two longer stretches that agree with each other
  gold         moves freely, but a value that differs from two agreeing
               neighbours for a single frame is a misread

It also marks which frames show the player's own board (`home_board`): the
HUD numbers are always the player's own, but while scouting or fighting away
the units and augment bar on screen belong to someone else.

Board and bench units (`units_raw` from scripts/read_units.py) are cleaned in
src/state/units.py: one-frame flickers and misreads smoothed out, the board
checked against the level, and the last planning board carried through combat.

Cleaning replaces values that break these rules and forward-fills gaps, so the
Macro Engine sees a complete, consistent sequence. The raw rows are not
modified; `filled` on each cleaned row lists which fields were not read
directly from that frame.
"""


from collections import Counter

from src.state.game_data import load_game_data
from src.state.units import clean_units
from src.vision.arena import home_board_rows, planning_rows

SCALARS = ("stage", "round_num", "gold", "hp", "level", "xp_current", "xp_needed", "streak")


def _monotonic(values: list, increasing: bool) -> list:
    """
    Keep the readings that form the best-supported monotonic sequence; drop the rest.

    A greedy "never go backwards" rule fails as soon as one misread is too
    extreme (HP read as 6 early on would reject every true reading after it).
    Instead, consecutive equal readings are grouped into runs, and the chain of
    runs that moves only in the allowed direction and covers the most frames
    wins. Misreads are short runs that don't fit the chain.
    """
    runs: list[list] = []   # [value, indices]
    for i, v in enumerate(values):
        if v is None:
            continue
        if runs and runs[-1][0] == v:
            runs[-1][1].append(i)
        else:
            runs.append([v, [i]])

    def fits(a, b) -> bool:
        return a <= b if increasing else a >= b

    # Heaviest chain ending at each run (weight = frames covered).
    best = [len(idx) for _, idx in runs]
    prev = [-1] * len(runs)
    for j in range(len(runs)):
        for i in range(j):
            if fits(runs[i][0], runs[j][0]) and best[i] + len(runs[j][1]) > best[j]:
                best[j] = best[i] + len(runs[j][1])
                prev[j] = i

    out = [None] * len(values)
    j = max(range(len(runs)), key=best.__getitem__) if runs else -1
    while j != -1:
        for i in runs[j][1]:
            out[i] = values[i]
        j = prev[j]
    return out


def _despike(values: list) -> list:
    """Drop a value that differs from two agreeing neighbours."""
    out = list(values)
    for i in range(1, len(values) - 1):
        a, v, b = values[i - 1], values[i], values[i + 1]
        if v is not None and a is not None and a == b and v != a:
            out[i] = None
    return out


def _drop_blips(values: list) -> list:
    """
    Drop short runs of a value that interrupt a longer stretch of another value.

    HP read as 6 for a few frames in the middle of a long run of 100s is a
    misread: the readings before and after agree with each other and both
    last longer. A real change (damage or a heal) is followed by the new
    value, not by a return to the old one.
    """
    runs: list[list] = []   # [value, indices]
    for i, v in enumerate(values):
        if v is None:
            continue
        if runs and runs[-1][0] == v:
            runs[-1][1].append(i)
        else:
            runs.append([v, [i]])

    out = list(values)
    changed = True
    while changed:
        changed = False
        for j in range(1, len(runs) - 1):
            before, blip, after = runs[j - 1], runs[j], runs[j + 1]
            if before[0] == after[0] and len(blip[1]) < min(len(before[1]), len(after[1])):
                for i in blip[1]:
                    out[i] = None
                before[1].extend(after[1])
                del runs[j:j + 2]
                changed = True
                break
    return out


def _forward_fill(values: list) -> list:
    out, last = [], None
    for v in values:
        last = v if v is not None else last
        out.append(last)
    return out


AUGMENT_SLOTS = 3
AUGMENT_PRESENT = 0.5     # a slot counts as picked from the first frame after which at least this
AUGMENT_CONFIRM = 2       # share of the readings that show the bar show the slot, and at least
                          # this many readings show it (one phantom icon is not a pick)


def game_augments(rows: list[dict], home: list[bool], label: dict | None = None) -> tuple[list, list, list]:
    """
    The game's augments, decided once per game instead of per frame.

    A game has exactly three, picked at 2-1, 3-2 and 4-2 and never changed,
    each in its own place in the bar (left to right in pick order). Per bar
    position: the slot is picked from the first own-board frame after which
    at least AUGMENT_PRESENT of the readings that show the bar (an empty
    reading is a hidden bar, not a lost augment) show an icon there, with
    at least AUGMENT_CONFIRM readings in all; its name and tier are the most
    common readings from then on. A per-game label from
    scripts/label_augments.py (data/processed/augment_labels.json) replaces the
    names and tiers.

    Returns (names, tiers, picked_at): picked_at is the row index from which
    each slot shows, or None if it was never seen.
    """
    readings = [(i, r.get("augments") or [], r.get("augment_tiers") or [])
                for i, r in enumerate(rows) if home[i]]
    names, tiers, picked = [], [], []
    for k in range(AUGMENT_SLOTS):
        has = [len(a) > k for _, a, _ in readings]
        bar = [len(a) > 0 for _, a, _ in readings]
        start = None
        # From the end backwards: share of bar-showing readings that show slot k.
        total, shown = 0, 0
        for j in range(len(readings) - 1, -1, -1):
            total += bar[j]
            shown += has[j]
            if has[j] and shown >= AUGMENT_CONFIRM and shown / total >= AUGMENT_PRESENT:
                start = j
        if start is None:
            break
        after = [(a[k], t[k] if k < len(t) else None) for _, a, t in readings[start:] if len(a) > k]
        name_votes = Counter(n for n, _ in after if n)
        tier_votes = Counter(t for _, t in after if t)
        names.append(name_votes.most_common(1)[0][0] if name_votes else None)
        tiers.append(tier_votes.most_common(1)[0][0] if tier_votes else None)
        picked.append(readings[start][0])
    if label:
        for k, n in enumerate(label.get("augments") or []):
            # "_unknown": an augment Hamza saw but couldn't read - keep the reader's guess
            if k < len(names) and n != "_unknown":
                names[k] = n
                if label.get("tiers") and k < len(label["tiers"]) and label["tiers"][k]:
                    tiers[k] = label["tiers"][k]
    return names, tiers, picked


def clean_game(rows: list[dict], augment_label: dict | None = None) -> list[dict]:
    """
    Return cleaned copies of one game's rows, in frame order.

    Each cleaned row has every SCALARS field filled where any earlier frame
    provided it, plus `filled`: the fields that were corrected or carried
    forward rather than read from that frame. augment_label: Hamza's labels
    for this game's augments (scripts/label_augments.py), if any.
    """
    rows = sorted(rows, key=lambda r: r["frame"])
    raw = {k: [r.get(k) for r in rows] for k in SCALARS}

    # Stage and round are one counter: filter and order them together.
    progress = [(s, r) if s is not None and r is not None else None
                for s, r in zip(raw["stage"], raw["round_num"])]
    progress = _monotonic(progress, increasing=True)

    checked = {
        "stage": [p[0] if p else None for p in progress],
        "round_num": [p[1] if p else None for p in progress],
        "level": _monotonic(raw["level"], increasing=True),
        "hp": _drop_blips(raw["hp"]),
        "gold": _despike(raw["gold"]),
        "xp_current": _despike(raw["xp_current"]),
        "xp_needed": _monotonic(raw["xp_needed"], increasing=True),
        "streak": _despike(raw["streak"]),
    }
    final = {k: _forward_fill(v) for k, v in checked.items()}

    home = home_board_rows(rows)
    planning = planning_rows(rows)
    rounds = list(zip(final["stage"], final["round_num"]))
    units, _ = clean_units(rows, final["level"], load_game_data(), rounds)

    # Augments: decided once per game, per bar position (game_augments);
    # readings off someone else's board are ignored.
    aug_names, aug_tiers, aug_picked = game_augments(rows, home, augment_label)

    cleaned = []
    for i, row in enumerate(rows):
        new = dict(row)
        filled = []
        new["home_board"] = home[i]
        new["planning"] = planning[i]
        new.pop("arena_sig", None)
        new.pop("units_raw", None)
        new.pop("board_counter", None)
        new["board"], new["bench"] = units[i]["board"], units[i]["bench"]
        if not units[i]["read"] and (new["board"] or new["bench"]):
            filled.append("board")
        held = sum(1 for p in aug_picked if i >= p)
        new["augments"], new["augment_tiers"] = aug_names[:held], aug_tiers[:held]
        if new["augments"] != (row.get("augments") or []):
            filled.append("augments")
        for k in SCALARS:
            new[k] = final[k][i]
            if final[k][i] is not None and final[k][i] != raw[k][i]:
                filled.append(k)
        new["filled"] = filled
        cleaned.append(new)
    return cleaned


def quality_report(rows: list[dict], cleaned: list[dict]) -> dict:
    """Per-field read rate and correction counts for one game."""
    n = len(rows)
    with_bar = [r for r in rows if r.get("shop_visible")]
    report = {"frames": n, "bottom_bar_visible": len(with_bar), "fields": {}}
    by_frame = {r["frame"]: r for r in rows}
    for k in SCALARS:
        # stage and hp are readable whenever the stage bar is up; the rest need the bottom bar
        base = rows if k in ("stage", "round_num", "hp") else with_bar
        read = sum(1 for r in base if r.get(k) is not None)
        changed = sum(1 for c in cleaned
                      if by_frame[c["frame"]].get(k) is not None and c[k] != by_frame[c["frame"]][k])
        report["fields"][k] = {
            "read_rate": round(read / len(base), 3) if base else None,
            "misreads_corrected": changed,
        }
    _, report["units"] = clean_units(sorted(rows, key=lambda r: r["frame"]), [c["level"] for c in cleaned],
                                     load_game_data(), [(c["stage"], c["round_num"]) for c in cleaned])
    first, last = cleaned[0], cleaned[-1]
    report["first"] = {k: first[k] for k in ("stage", "round_num", "hp", "level")}
    report["last"] = {k: last[k] for k in ("stage", "round_num", "hp", "level")}
    return report
