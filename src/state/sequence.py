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
HUD numbers are the player's own while he is in the game, but while scouting or
fighting away the units and augment bar on screen belong to someone else.

After he is knocked out the stream often keeps showing another player's game
with a full HUD (spectating), and a game folder can start with the previous
game's last frames; `own_span` finds the stretch that is his one game.

Board and bench units (`units_raw` from scripts/read_units.py) are cleaned in
src/state/units.py: one-frame flickers and misreads smoothed out, the board
checked against the level, and the last planning board carried through combat.

Cleaning replaces values that break these rules and forward-fills gaps, so the
Macro Engine sees a complete, consistent sequence. The raw rows are not
modified; `filled` on each cleaned row lists which fields were not read
directly from that frame.
"""


import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from src.state.game_data import load_game_data
from src.state.units import clean_units
from src.vision.arena import home_board_rows, planning_rows
from src.vision.augments import MAX_AUGMENTS

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


AUGMENT_NAMES = Path(__file__).resolve().parents[2] / "assets" / "templates" / "augments_v2" / "names.json"
AUGMENT_ALIASES = AUGMENT_NAMES.with_name("aliases.json")   # same icon, two names -> one (build_augment_templates.py)


@lru_cache(maxsize=1)
def _augment_names() -> dict:
    try:
        return json.loads(AUGMENT_NAMES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def _augment_aliases() -> dict:
    try:
        return json.loads(AUGMENT_ALIASES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def canonical_augment(name: str | None) -> str | None:
    """One name for augments that share an icon in the game (Spreading Roots / Max Build, ...)."""
    if not name:
        return name
    from src.state.game_data import GameData
    folded = GameData.augment_label(name)
    return _augment_aliases().get(folded, folded)


def augment_name(template: str | None, tier: str | None) -> str | None:
    """
    An augment template's name from scripts/name_augment_templates.py: by template, or by
    template and tier where one template stands for two augments that differ only by tier
    colour (Silver / Gold Destiny). Unnamed templates keep their ID.
    """
    names = _augment_names()
    return canonical_augment(names.get(f"{template}@{tier}") or names.get(template) or template)


AUGMENT_SLOTS = MAX_AUGMENTS   # 3 picks, and a 4th seen in some games
AUGMENT_PRESENT = 0.5     # a slot counts as picked from the first frame after which at least this
AUGMENT_CONFIRM = 2       # share of the readings that show the bar show the slot, and at least
                          # this many readings show it (one phantom icon is not a pick)


def game_augments(rows: list[dict], home: list[bool], label: dict | None = None) -> tuple[list, list, list]:
    """
    The game's augments, decided once per game instead of per frame.

    A game has three picks, at 2-1, 3-2 and 4-2, sometimes a fourth augment
    from elsewhere; none ever changes, and each keeps its own place in the
    bar (left to right in the order received). Per bar
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
        after = [(augment_name(a[k], t[k] if k < len(t) else None), t[k] if k < len(t) else None)
                 for _, a, t in readings[start:] if len(a) > k]
        name_votes = Counter(n for n, _ in after if n)
        tier_votes = Counter(t for _, t in after if t)
        names.append(name_votes.most_common(1)[0][0] if name_votes else None)
        tiers.append(tier_votes.most_common(1)[0][0] if tier_votes else None)
        picked.append(readings[start][0])
    if label:
        for k, n in enumerate(label.get("augments") or []):
            # "_unknown": an augment Hamza saw but couldn't read - keep the reader's guess
            if k < len(names) and n != "_unknown":
                names[k] = canonical_augment(n)
                if label.get("tiers") and k < len(label["tiers"]) and label["tiers"][k]:
                    tiers[k] = label["tiers"][k]
    return names, tiers, picked


SETTLE = 3          # a reading counts for own_span once it repeats this many times in a row
HP_JUMP = 15        # settled HP this far above his lowest settled HP so far = another player's HUD
LASTING = 20        # a change must hold for most of the next this-many readings of that field
STAGE_BACK = 10     # settled round counter (stage*10 + round) this far below its highest: a whole stage back
                    # (OCR swaps 3/5 and settles a 6-3 among 6-5s: SZZrxLQq_gI/game_04, wszJroIjlMs/game_06)


def _settled(rows: list[dict], get) -> list[tuple[int, object]]:
    """(row index, value) of readings that repeat SETTLE times in a row (None skipped):
    a single misread never counts."""
    out, run, last = [], 0, object()
    for i, r in enumerate(rows):
        v = get(r)
        if v is None:
            continue
        run = run + 1 if v == last else 1
        last = v
        if run >= SETTLE:
            out.append((i, v))
    return out


def _round(r: dict) -> float | None:
    s, n = r.get("stage"), r.get("round_num")
    return s * 10 + n if s and n else None


def own_span(rows: list[dict]) -> tuple[int, int, str | None]:
    """
    (start, end, reason): rows[start:end] are his own game (rows in frame order).

    Start: after the last reset to stage 1 (a folder can begin with the previous
    game's last frames: stage 6-5 then 1-1, HP back to 100).
    End: before the HUD stops being his, the first of (settled readings only)
      - HP 15+ above his lowest HP so far (another player's HUD; heals are a few HP a round),
      - the stage going back a whole stage (a replay or another game; a round or two back is OCR),
      - the level going down (levels 1-10 only: a misread 326 is not a level).
    A change must last (most of the next LASTING readings): a misread can settle for 3 frames.
    The cut is after his last settled reading of that value before the change.
    Found on 2026-11-23 in 17 of 202 games (TRYn_hEfx7Y/game_10: HP 7 at 4-4, then 38 at
    level 8 from 5-5 to 6-5, another player's augments on the bar).
    """
    start = 0
    rounds = _settled(rows, _round)
    hi = None
    for i, v in rounds:
        if hi is not None and v <= 13 < hi:              # back to stage 1 after a later stage
            start = i - SETTLE + 1
            while start > 0 and _round(rows[start - 1]) in (None, v):
                start -= 1
            hi = None
        hi = v if hi is None else max(hi, v)
    own = rows[start:]

    cuts = []

    def first_change(get, changed, update, label):
        """The first settled reading that `changed` from the reference, where the change lasts
        (most of the next LASTING readings agree); a passing misread is skipped, not taken as
        the new reference (--ppQShtXow/game_08 read level 5 three frames running at level 8)."""
        ref, seq = None, _settled(own, get)
        for k, (i, v) in enumerate(seq):
            if ref is not None and changed(v, ref):
                after = [x for x in (get(r) for r in own[i:]) if x is not None][:LASTING]
                if sum(changed(x, ref) for x in after) >= 0.6 * len(after):
                    prev = max((j for j, w in seq[:k] if not changed(w, ref)), default=i - 1)
                    cuts.append((prev + 1, label(ref, v)))
                    return
                continue
            ref = update(ref, v)

    first_change(lambda r: r.get("hp"), lambda v, low: v >= low + HP_JUMP,
                 lambda low, v: v if low is None else min(low, v), lambda a, b: f"HP {a:.0f} -> {b:.0f}")
    first_change(_round, lambda v, hi: v <= hi - STAGE_BACK,
                 lambda hi, v: v if hi is None else max(hi, v), lambda a, b: f"stage {a / 10:.1f} -> {b / 10:.1f}")
    first_change(lambda r: r.get("level") if r.get("level") and r["level"] <= 10 else None,
                 lambda v, hi: v < hi, lambda hi, v: v if hi is None else max(hi, v),
                 lambda a, b: f"level {a:.0f} -> {b:.0f}")
    if not cuts:
        return start, len(rows), None
    end, reason = min(cuts)
    return start, start + end, reason


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
    # Augments: decided once per game, per bar position (game_augments);
    # readings off someone else's board are ignored.
    aug_names, aug_tiers, aug_picked = game_augments(rows, home, augment_label)
    held_augments = [[a for a, p in zip(aug_names, aug_picked) if i >= p] for i in range(len(rows))]
    units, _ = clean_units(rows, final["level"], load_game_data(), rounds, checked["gold"], held_augments)

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
    ordered = sorted(rows, key=lambda r: r["frame"])
    _, report["units"] = clean_units(ordered, [c["level"] for c in cleaned],
                                     load_game_data(), [(c["stage"], c["round_num"]) for c in cleaned],
                                     _despike([r.get("gold") for r in ordered]),
                                     [c.get("augments") or [] for c in cleaned])
    first, last = cleaned[0], cleaned[-1]
    report["first"] = {k: first[k] for k in ("stage", "round_num", "hp", "level")}
    report["last"] = {k: last[k] for k in ("stage", "round_num", "hp", "level")}
    return report
