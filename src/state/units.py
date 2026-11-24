"""
Turn single-frame unit readings (units_raw, from scripts/read_units.py) into
a cleaned board and bench for every frame of a game.

Single frames are noisy: a unit's health bar flickers behind an effect, the
classifier swaps two look-alikes for one frame, the star badge is hidden by
a tooltip. The sequence fixes most of it:

1. Frame weight. Dedup deleted every frame that looked the same as the one
   before it (gold, shop, bench, item column and stage text), so a kept frame
   stands for all the seconds until the next kept one. A frame that stood for
   10 s was on screen, unchanged, for 20 captured frames: nothing was moving,
   so it is the most trustworthy kind of reading. Each frame's weight is that
   time, capped at MAX_STABLE_SECONDS. (The board itself is not one of the
   watched regions: moving a unit between hexes alone does not keep a frame.)

2. Per position (each hex and bench slot), the readings over the planning
   frames form a sequence: "Ashe, Ashe, (empty), Ashe" or "Ashe, Jinx, Ashe".
   A short run that interrupts two longer runs agreeing with each other is
   noise and takes their value: the unit never left, or was never something
   else. "Longer" is by weight (time x classifier confidence), and only runs
   of up to MAX_BLIP_FRAMES frames are overwritten, so a real swap that
   lasted a while stays. A unit read with confidence >= P_SURE is never
   overwritten by "empty": a sure reading is strong evidence the unit was
   there, a missing bar is weak (2026-11-24: on 1,596 held-out labelled
   crops, emptying such readings removed 34 correctly read units; many sat
   between frames where the same unit's bar was hidden, and the empty
   frames outweighed it).

3. Star level and items are smoothed the same way within each stretch of the
   same unit standing on the same spot, and missing readings (badge hidden)
   are filled from the rest of that stretch.

4. Team size check. The game's own counter ("units / team size" in the
   arena, src/vision/board_counter.py) is used first: the team size per
   round (two agreeing readings, between level and level + 3), and the unit
   count per frame (agreeing with nearby frames). Where it can't be read,
   the board can hold at most level + bonus team slots from the set's own
   data (GameData):
     - trait and augment units (Stonebark Tree, Training Dummy, ...) take no
       slot (on knB6dE8Rbgg, counting them put the board over the level on
       68 frames, leaving them out on 2); Elder Dragon takes 2;
     - the bonus comes from active traits (Riftbeast 10: +2) and team-size
       items held by units (Tactician's Crown/Cape/Shield: +1).
   Augments that add team size (New Recruit, Cursed Crown, ...) are not
   known (augment icons only have IDs), so their extra unit gets dropped:
   ev6dfSvJuJ8 game_03 shows "10/10" at level 9 from stage 5 on. When more
   is read than fits, the least supported units are dropped; the count of
   such frames goes into the quality report.

5. Carry forward. When the counter (agreeing over nearby frames) shows more
   units than were read, units seen on the board in the last CARRY_LOOKBACK
   read frames that have since vanished are put back, most recently seen
   first, until the count matches. Not a unit that just moved (its champion
   is already counted on its new hex), and never more than the counter says.
   Carried units have "carried": true.

6. Stretches where the board can't change (Hamza, 2026-11-24). Between two
   read frames of one round with the same gold and the same bench (by name),
   no unit can have joined or left the board: a unit goes on or off the
   board through the bench, and a sale or a purchase changes gold. (Units
   can still swap hexes and take items.) Such frames form a stretch, and
   after step 2 each stretch gets one set of board units: every copy of a
   champion is voted over the stretch's frames (weight x confidence for,
   EMPTY_WEIGHT x weight against), and where the counter was read in the
   stretch (two agreeing readings) the best supported copies are taken up
   to its count. Each frame then drops the copies it has too many of (least
   supported first) and gets the missing ones at the hex they stood on in
   the nearest frame of the stretch ("filled"). A champion with one copy
   gets one star level over the stretch (stars only change by buying the
   third copy, which costs gold). Gold unread (shop bar hidden) counts as
   unchanged only when the readings around agree. On 7 re-read games: the
   board changed between such frames in 14.9% of pairs before, 2.3% after;
   with the counter hidden on a quarter of frames, the board count matched
   it on 60.2% -> 68.8% there; held-out labelled crops 89.6% -> 89.8%.

   Stacks (Hamza): when a frame of a stretch lacks a unit whose hex holds
   another unit read there, and that other unit's own hex is empty, one was
   dragged over the other: both go back to their own hexes ("filled").

7. Doubted champions (Hamza, 2026-11-24). A champion gets onto the board or
   bench by being bought, so in a round where it was neither in the shop
   (this round or the last) nor read surely (p >= SKEPTIC_SURE, this round
   or the last), a reading of it is probably a misread: its confidence is
   multiplied by SKEPTIC, so steps 2-6 trust it less ("doubted"). When the
   reading carries the model's next guesses (`alt`, read_units.py since
   2026-11-24), it takes the first one that could be there instead. Not in
   rounds that hand out units (free_unit_round: stage 1, carousels, PvE
   orb rounds, augment rounds), and never for trait/augment units or
   `_unknown`.

Frames that were not read (combat, away fights, scouting) carry the last
planning board forward: the units only move during combat and come back.
"""

from collections import Counter, defaultdict

MAX_STABLE_SECONDS = 10.0
MAX_BLIP_FRAMES = 2
FPS = 2.0                      # frame numbers are at 2 fps (frame / 2 = seconds)
EMPTY = None
# "Nothing here" is weaker evidence than a unit seen: spell effects and
# overlapping units hide health bars for a frame far more often than a
# unit appears from nowhere. An empty reading counts this much of a sure one.
EMPTY_WEIGHT = 0.5
# A unit reading at least this sure is not erased as a flicker of "empty"
# (it can still be renamed by agreeing neighbours). 0.5-0.8 scored the same.
P_SURE = 0.8


def frame_weights(frames: list[int], max_seconds: float = MAX_STABLE_SECONDS) -> list[float]:
    """Seconds each kept frame stood for (until the next kept frame), capped."""
    out = []
    for i, f in enumerate(frames):
        gap = (frames[i + 1] - f) / FPS if i + 1 < len(frames) else 1 / FPS
        out.append(min(max(gap, 1 / FPS), max_seconds))
    return out


def smooth(labels: list, weights: list[float], max_blip: int = MAX_BLIP_FRAMES,
           sure: list[bool] | None = None) -> list:
    """
    Overwrite short runs that interrupt two longer runs of the same label.

    labels: one label per reading (any hashable; None is a label too, e.g. "empty").
    weights: how much each reading counts. A run is overwritten when it has at
    most `max_blip` readings and no more total weight than each neighbour run.
    sure: per reading, True where it may not be overwritten by EMPTY (a sure unit reading).
    """
    runs: list[list] = []          # [label, [indices], weight]
    for i, (lab, w) in enumerate(zip(labels, weights)):
        if runs and runs[-1][0] == lab:
            runs[-1][1].append(i)
            runs[-1][2] += w
        else:
            runs.append([lab, [i], w])

    out = list(labels)
    changed = True
    while changed:
        changed = False
        # Weakest blip first, so one flicker doesn't decide its neighbour's fate.
        candidates = sorted(range(1, len(runs) - 1), key=lambda j: runs[j][2])
        for j in candidates:
            before, blip, after = runs[j - 1], runs[j], runs[j + 1]
            if (before[0] == after[0] and len(blip[1]) <= max_blip
                    and blip[2] <= min(before[2], after[2])
                    and not (before[0] is EMPTY and sure is not None and any(sure[i] for i in blip[1]))):
                for i in blip[1]:
                    out[i] = before[0]
                before[1].extend(blip[1] + after[1])
                before[2] += blip[2] + after[2]
                del runs[j:j + 2]
                changed = True
                break
    return out


def _fill_within(values: list, groups: list) -> list:
    """Fill None from the nearest earlier (else later) value in the same group."""
    out = list(values)
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    for idx in by_group.values():
        last = None
        for i in idx:
            last = out[i] if out[i] is not None else last
            out[i] = last
        last = None
        for i in reversed(idx):
            last = out[i] if out[i] is not None else last
            out[i] = last
    return out


def _stretches(names: list) -> list:
    """An id per stretch of the same name (consecutive readings), so stars/items are smoothed within it."""
    ids, k = [], 0
    for i, n in enumerate(names):
        if i and n != names[i - 1]:
            k += 1
        ids.append(k)
    return ids


SKEPTIC = 0.5             # confidence factor of a champion that had no way onto the board (step 7)
SKEPTIC_SURE = 0.9        # a reading this sure shows the champion is there (board 541/543 right at 0.9)
ALT_MIN = 0.1             # a next guess this likely may replace a doubted reading
STACKS = True             # step 6: put a unit dragged over another back on its hex


def free_unit_round(key) -> bool:
    """Rounds that can hand out units without a purchase: all of stage 1 (carousel,
    PvE drops), carousels (x-4), PvE orb rounds (x-7) and augment rounds (2-1, 3-2, 4-2).
    An unknown round counts as one (nothing is doubted then)."""
    if key is None or key[0] is None or key[1] is None:
        return True
    s, r = int(key[0]), int(key[1])
    return s == 1 or r in (4, 7) or (s, r) in ((2, 1), (3, 2), (4, 2))


def accounted(rows: list[dict], rounds: list, always: set) -> list[set | None]:
    """
    Per row (step 7): the champions that had a way onto the board or bench:
    in the shop this round or the last, read with p >= SKEPTIC_SURE this
    round or the last (from a round handing out units: read with p >= 0.5 on
    2+ frames there), or in `always`. None in a round handing out units
    (anything could be there). On 1,596 held-out labelled crops (2026-11-24):
    outside those rounds 26 readings doubted, 22 of them misreads (of 100
    misreads there); exempting the round after each such round as well left
    two thirds of all readings undoubted.
    """
    order, shop, sure, seen = [], defaultdict(set), defaultdict(set), defaultdict(Counter)
    for r, key in zip(rows, rounds):
        if not order or order[-1] != key:
            order.append(key)
        shop[key] |= {n for n in (r.get("shop_slots") or []) if n}
        for u in r.get("units_raw") or []:
            if u.get("name"):
                if u.get("p", 0) >= SKEPTIC_SURE:
                    sure[key].add(u["name"])
                if u.get("p", 0) >= 0.5:
                    seen[key][u["name"]] += 1
    ok, prev = {}, None
    for key in order:
        if key in ok:
            continue
        if free_unit_round(key):
            ok[key] = None
        else:
            ok[key] = shop[key] | sure[key] | always
            if prev is not None:
                ok[key] |= shop[prev] | sure[prev]
                if free_unit_round(prev):
                    ok[key] |= {n for n, c in seen[prev].items() if c >= 2}
        prev = key
    return [ok[key] for key in rounds]


def _doubt(units: list[dict], ok: set | None, stats: dict) -> list[dict]:
    """A frame's readings with champions that had no way there renamed to a next guess that
    had one (if the reading carries `alt`), else trusted SKEPTIC times less."""
    if ok is None:
        return units
    out = []
    for u in units:
        if u.get("name") is None or u["name"] in ok:
            out.append(u)
            continue
        alt = next(((n, q) for n, q in u.get("alt") or [] if n in ok and q >= ALT_MIN), None)
        if alt:
            out.append(dict(u, name=alt[0], p=round(alt[1], 3), renamed=u["name"]))
            stats["doubted_renamed"] += 1
        else:
            out.append(dict(u, p=round(u["p"] * SKEPTIC, 3), doubted=True))
            stats["doubted"] += 1
    return out


# Team size for the cap (step 4): "computed" from the state (GameData.team_size: level, traits,
# augments, team-size items; Hamza, 2026-11-24), "counter" (the arena's number per round, else
# level + traits + board-held items: before 2026-11-24), "counter_first" (counter where read).
TEAM_SIZE_FROM = "computed"
KEEP_ITEM_FRAMES = 3      # a team-size item seen on this many frames of a round is owned from then on
# Where the counter's unit count isn't read, carry vanished units back up to the computed team size
# (the board is full on 84% of frames with a trusted counter, one short on 15%). On 7 re-read games
# with the counter hidden on a quarter of frames, the board count matched it there 71.2% -> 95.3%
# (4.5% one too many); held-out crops 90.0% -> 90.3%.
FILL_TO_TEAM_SIZE = True
COUNTER_MAX_BONUS = 3     # team size read above level + this is a misread
CARRY_LOOKBACK = 6        # read frames back a vanished unit may be carried from (counter permitting)
COUNTER_WINDOW = 2        # unit count: read frames either side that are compared
COUNTER_AGREE = 2         # ...and how many readings (incl. the frame's own) must agree


def _counter_team_sizes(rows, read, levels, rounds) -> dict[int, int]:
    """
    Team size per read frame from the board counter: per round, the most
    common reading, if at least two readings agree and it is between the
    level and level + COUNTER_MAX_BONUS (a reading below the level is
    impossible). The team size only changes between rounds.
    """
    votes: dict = defaultdict(list)
    for k, i in enumerate(read):
        c, level = rows[i].get("board_counter"), levels[i]
        if c and c[1] is not None and level and level <= c[1] <= level + COUNTER_MAX_BONUS:
            votes[rounds[i]].append(c[1])
    best = {}
    for key, vs in votes.items():
        value = max(set(vs), key=vs.count)
        if key is not None and vs.count(value) >= 2:
            best[key] = value
    return {k: best[rounds[i]] for k, i in enumerate(read) if rounds[i] in best}


def _counter_units(rows, read, rounds) -> dict[int, int]:
    """
    Units on the board from the counter, per read frame: only when the
    reading agrees with at least COUNTER_AGREE readings (its own included)
    among the nearby read frames of the same round. One frame alone is not
    trusted: monsters' health bars and effects can cover a digit.
    """
    vals = [(rows[i].get("board_counter") or [None])[0] for i in read]
    out = {}
    for k, i in enumerate(read):
        v = vals[k]
        if v is None:
            continue
        near = [vals[j] for j in range(max(0, k - COUNTER_WINDOW), min(len(read), k + COUNTER_WINDOW + 1))
                if rounds[read[j]] == rounds[i]]
        if near.count(v) >= COUNTER_AGREE:
            out[k] = v
    return out


def _flags(u: dict) -> dict:
    """How sure a cleaned unit is, kept only when set: carried (put back on the
    counter's word), filled (not read as this unit on this frame), loose (its
    bar was not over its hex), doubted (no way there, step 7)."""
    return {k: True for k in ("carried", "filled", "loose", "doubted") if u.get(k)}


def _keep_loose(units: list[dict], counter: int | None, stats: dict) -> list[dict]:
    """
    A frame's readings with its loose ones (bars no hex claimed: a unit being
    dragged, or one off its hex) kept only as far as the board counter says
    the board is short: a unit dragged from the bench floats over the board
    but the game doesn't count it until it is dropped. Without a trusted
    count, loose readings are left out. Most confident first.
    """
    loose = [u for u in units if u.get("loose")]
    if not loose:
        return units
    fixed = [u for u in units if not u.get("loose")]
    short = 0
    if counter is not None:
        on_board = sum(1 for u in fixed if u["at"][0] == "h" and not u.get("covered"))
        short = max(0, counter - on_board)
    keep = sorted(loose, key=lambda u: -u.get("p", 0))[:short]
    stats["loose_kept"] += len(keep)
    stats["loose_dropped"] += len(loose) - len(keep)
    return fixed + keep


_UNREAD = object()


def _bracket(values: list) -> list:
    """
    An unread value (None) takes the value read around it when the readings
    before and after agree; otherwise a marker equal to nothing, so a change
    somewhere in an unread stretch is not missed (no telling where it was).
    """
    out = list(values)
    for i, v in enumerate(values):
        if v is None:
            a = next((values[j] for j in range(i - 1, -1, -1) if values[j] is not None), _UNREAD)
            b = next((values[j] for j in range(i + 1, len(values)) if values[j] is not None), _UNREAD)
            out[i] = a if (a is not _UNREAD and a == b) else object()
    return out


def board_stretches(cleaned: list[dict], read: list[int], rounds: list, gold: list) -> list[int]:
    """
    A stretch id per read frame (step 6): consecutive read frames of one
    round with the same gold and the same bench (champion names) share one.
    cleaned: per read frame {position: unit}; rounds and gold per row.
    """
    g = _bracket([gold[i] for i in read])
    bench = [tuple(sorted(u["name"] for at, u in c.items() if at[0] == "s")) for c in cleaned]
    out, s = [], 0
    for k in range(len(read)):
        if k and not (rounds[read[k]] == rounds[read[k - 1]] and g[k] == g[k - 1] and bench[k] == bench[k - 1]):
            s += 1
        out.append(s)
    return out


def _stretch_board(cleaned: list[dict], ks: list[int], w: list[float], counts: dict, slots) -> Counter:
    """The champions (with copies) on the board throughout one stretch of read frames ks."""
    per = {k: defaultdict(list) for k in ks}          # frame -> champion -> confidences, best first
    for k in ks:
        for at, u in cleaned[k].items():
            if at[0] == "h":
                per[k][u["name"]].append(u["p"])
        for c in per[k]:
            per[k][c].sort(reverse=True)
    cand = []
    for c in {c for k in ks for c in per[k]}:
        for n in range(1, 1 + max(len(per[k][c]) for k in ks)):
            yes = sum(w[k] * per[k][c][n - 1] for k in ks if len(per[k][c]) >= n)
            no = sum(w[k] * EMPTY_WEIGHT for k in ks if len(per[k][c]) < n)
            cand.append((yes - no, c))
    cand.sort(key=lambda t: -t[0])
    target = None
    read_counts = [counts[k] for k in ks if counts.get(k) is not None]
    if read_counts:
        value, n = Counter(read_counts).most_common(1)[0]
        target = value if n >= 2 else None
    chosen, used = Counter(), 0
    for score, c in cand:
        need = slots(c)
        if target is None or need == 0:
            ok = score > 0
        else:                       # up to the counter's count, seen copies only
            ok = used + need <= target and (score > 0 or used < target)
        if ok:
            chosen[c] += 1
            used += need
    return chosen


def _reconcile_stretches(cleaned: list[dict], stretch: list[int], w: list[float], counts: dict,
                         slots, stats: dict) -> None:
    """Step 6: give every frame of a stretch the stretch's board units (in place)."""
    by = defaultdict(list)
    for k, s in enumerate(stretch):
        by[s].append(k)
    for ks in by.values():
        if len(ks) < 2:
            continue
        before = {k: [(at, u) for at, u in cleaned[k].items() if at[0] == "h"] for k in ks}
        chosen = _stretch_board(cleaned, ks, w, counts, slots)
        for k in ks:
            units = cleaned[k]
            have = defaultdict(list)
            for at, u in before[k]:
                have[u["name"]].append(at)
            for c, ats in have.items():
                for at in sorted(ats, key=lambda a: units[a]["support"])[:max(len(ats) - chosen[c], 0)]:
                    del units[at]
                    stats["stretch_removed"] += 1
            for c, n in chosen.items():
                miss = n - sum(1 for at in units if at[0] == "h" and units[at]["name"] == c)
                for j in sorted(ks, key=lambda j: abs(j - k)):
                    if miss <= 0:
                        break
                    for at, u in before[j]:
                        if miss <= 0 or u["name"] != c or u.get("filled"):
                            continue
                        if at not in units:
                            units[at] = dict(u, filled=True, support=u["support"] * EMPTY_WEIGHT)
                            miss -= 1
                            stats["stretch_filled"] += 1
                        elif STACKS and units[at]["name"] != c:
                            # another unit on c's hex whose own hex (in frame j) is empty: one
                            # dragged over the other (Hamza); both back to their own hexes
                            d = units[at]
                            homes = [a for a, v in before[j] if v["name"] == d["name"] and a not in units]
                            if homes:
                                units[homes[0]] = dict(d, filled=True)
                                units[at] = dict(u, filled=True, support=u["support"] * EMPTY_WEIGHT)
                                miss -= 1
                                stats["stretch_stacked"] += 1
        for c, n in chosen.items():
            if n != 1:
                continue
            votes = Counter()
            for k in ks:
                for at, u in before[k]:
                    if u["name"] == c and u["star"] and not u.get("filled"):
                        votes[u["star"]] += w[k] * u["p"]
            if not votes:
                continue
            star = votes.most_common(1)[0][0]
            for k in ks:
                for at, u in cleaned[k].items():
                    if at[0] == "h" and u["name"] == c and u["star"] != star:
                        cleaned[k][at] = dict(u, star=star)
                        stats["stretch_stars"] += 1


def clean_units(rows: list[dict], levels: list, game=None, rounds: list | None = None,
                gold: list | None = None, augments: list | None = None) -> tuple[list[dict], dict]:
    """
    Cleaned board and bench for one game's rows (in frame order).

    Args:
        rows: raw state rows, sorted by frame; read frames have `units_raw`
            (and `board_counter`, the game's own "units / team size").
        levels: the cleaned player level for each row (None when unknown).
        game: GameData, for team slots and team-size bonuses (see step 4).
            Without it every unit takes one slot and there are no bonuses.
        rounds: the cleaned (stage, round) of each row, to group counter
            readings by round. Without it the counter is not used.
        gold: the gold read on each row (None where unread; not carried
            forward), for step 6. Without it step 6 is skipped.
        augments: the augment names held on each row (cleaned), for the
            team size (team-size augments). Without it none are counted.

    Returns:
        (per-row {"board": [...], "bench": [...], "read": bool}, stats)
        Board units: {"name", "star", "items", "hex": [row, col], "p"};
        bench units: {"name", "star", "items", "slot", "p"}.
        `p` is the classifier's confidence on that frame, or the mean over the
        unit's stretch when the reading was filled in.
    """
    weights_all = frame_weights([r["frame"] for r in rows])
    read = [i for i, r in enumerate(rows) if r.get("units_raw") is not None]
    stats = {"frames_read": len(read), "units_filled": 0, "units_removed": 0,
             "names_changed": 0, "over_level_frames": 0, "units_dropped_level": 0,
             "team_size_bonus_frames": 0, "team_size_from_counter": 0, "units_from_counter": 0,
             "counter_more_than_read": 0, "units_carried": 0, "covered_filled": 0,
             "loose_kept": 0, "loose_dropped": 0,
             "stretch_removed": 0, "stretch_filled": 0, "stretch_stars": 0, "stretch_stacked": 0,
             "doubted": 0, "doubted_renamed": 0}
    rounds = rounds or [None] * len(rows)
    out = [{"board": [], "bench": [], "read": False} for _ in rows]
    if not read:
        return out, stats

    w = [weights_all[i] for i in read]
    counts = _counter_units(rows, read, rounds)
    always = {"_unknown"} | set(game.extra_units if game is not None else ())
    ok = accounted(rows, rounds, always) if SKEPTIC < 1 and rounds[0] is not None else [None] * len(rows)
    raw = [{u["at"]: u for u in _keep_loose(_doubt(rows[i]["units_raw"], ok[i], stats), counts.get(k), stats)}
           for k, i in enumerate(read)]
    positions = sorted({at for frame in raw for at in frame})

    # Smoothed reading per (read frame, position).
    cleaned: list[dict] = [{} for _ in read]
    for at in positions:
        seen = [frame.get(at) for frame in raw]
        # Under a tooltip: no evidence either way. Left out of the smoothing
        # and given the value of the nearest reading before (else after).
        hidden = [bool(u and u.get("covered")) for u in seen]
        known = [k for k in range(len(seen)) if not hidden[k]]
        names = [seen[k]["name"] if seen[k] else EMPTY for k in known]
        conf_known = [w[k] * (seen[k]["p"] if seen[k] else EMPTY_WEIGHT) for k in known]
        smoothed = smooth(names, conf_known,
                          sure=[bool(seen[k]) and seen[k].get("p", 0) >= P_SURE for k in known])
        names2: list = [EMPTY] * len(seen)
        source = list(range(len(seen)))          # which reading each frame's value comes from
        for j, k in enumerate(known):
            names2[k] = smoothed[j]
        for k in range(len(seen)):
            if hidden[k]:
                prev = [j for j in known if j < k]
                nxt = [j for j in known if j > k]
                src = prev[-1] if prev else (nxt[0] if nxt else None)
                if src is not None:
                    names2[k], source[k] = names2[src], src
                stats["covered_filled"] += names2[k] is not EMPTY
        seen = [(seen[source[k]] if source[k] != k else None) if hidden[k] else seen[k]
                for k in range(len(seen))]
        conf = [w[k] * (seen[k]["p"] if seen[k] and "p" in seen[k] else EMPTY_WEIGHT) for k in range(len(seen))]
        groups = _stretches(names2)
        stars = [u["star"] if u and u["name"] == n else None for u, n in zip(seen, names2)]
        items = [tuple(u["items"]) if u and u["name"] == n else None for u, n in zip(seen, names2)]
        stars = _fill_within(smooth(stars, conf), groups)
        items = _fill_within(smooth(items, conf), groups)
        ps: dict[int, list[float]] = defaultdict(list)
        for u, n, g in zip(seen, names2, groups):
            if u and u["name"] == n:
                ps[g].append(u["p"])
        for k, (u, n) in enumerate(zip(seen, names2)):
            if n is EMPTY:
                if u:
                    stats["units_removed"] += 1
                continue
            if u is None:
                stats["units_filled"] += 1
            elif u["name"] != n:
                stats["names_changed"] += 1
            p = u["p"] if u and u["name"] == n else round(sum(ps[groups[k]]) / max(len(ps[groups[k]]), 1), 3)
            cleaned[k][at] = {"name": n, "star": stars[k], "items": list(items[k] or []), "p": p,
                              "support": conf[k],
                              # not read on this frame as this unit: a flicker, a misread
                              # overwritten, or hidden under a tooltip
                              "filled": u is None or hidden[k] or u["name"] != n,
                              "loose": bool(u and u.get("loose") and not hidden[k]),
                              "doubted": bool(u and u.get("doubted") and u["name"] == n)}

    slots = game.slots if game is not None else (lambda name: 1)
    if gold is not None and rounds[0] is not None:
        _reconcile_stretches(cleaned, board_stretches(cleaned, read, rounds, gold), w, counts, slots, stats)
    team_sizes = _counter_team_sizes(rows, read, levels, rounds)
    counts = _counter_units(rows, read, rounds)
    last_seen: dict[str, tuple[dict, int]] = {}      # board hex -> (unit, read-frame index last seen)
    recent_counts: list[Counter] = []                 # champion copies on the board, per read frame
    recent_bench: list[Counter] = []                  # ...and on the bench
    kept_items: Counter = Counter()                   # team-size items known to be owned
    round_items: Counter = Counter()                  # (item, copies) -> frames this round
    this_round = None
    for k, i in enumerate(read):
        units = cleaned[k]
        board = sorted((at for at in units if at[0] == "h"), key=lambda at: -units[at]["support"])
        level = levels[i]
        cap = None
        computed = None
        if level and game is not None:
            if rounds[i] != this_round:
                for (item, n), frames in round_items.items():
                    if frames >= KEEP_ITEM_FRAMES:
                        kept_items[item] = max(kept_items[item], n)
                round_items.clear()
                this_round = rounds[i]
            board_units = [units[at] for at in board]
            elsewhere = [x for at, u in units.items() if at[0] == "s" for x in u["items"] or []]
            elsewhere += [x for x in rows[i].get("bench_items") or [] if x]
            here = Counter(x for x in [x for u in board_units for x in u["items"] or []] + elsewhere
                           if x in game.team_size_items)
            for item, n in here.items():
                for m in range(1, n + 1):
                    round_items[(item, m)] += 1
            computed = game.team_size(level, board_units, elsewhere, augments[i] if augments else (), kept_items)
        elif level:
            computed = int(level)
        if TEAM_SIZE_FROM != "computed" and team_sizes.get(k) is not None:
            cap = team_sizes[k]
            stats["team_size_from_counter"] += 1
        elif TEAM_SIZE_FROM == "counter" and level:
            bonus = game.team_size_bonus([units[at] for at in board]) if game is not None else 0
            cap = int(level) + bonus
            stats["team_size_bonus_frames"] += bonus > 0
        elif computed is not None:
            cap = computed
            stats["team_size_bonus_frames"] += computed > int(level)
        if counts.get(k) is not None and (cap is None or counts[k] <= cap):
            cap = counts[k]                          # the game's own unit count is tighter
            stats["units_from_counter"] += 1
            used_now = sum(slots(units[at]["name"]) for at in board)
            stats["counter_more_than_read"] += used_now < cap
        if cap is not None and sum(slots(units[at]["name"]) for at in board) > cap:
            stats["over_level_frames"] += 1
            used = 0
            for at in board:                         # best supported first
                need = slots(units[at]["name"])
                if used + need > cap:
                    del units[at]
                    stats["units_dropped_level"] += 1
                else:
                    used += need

        # The counter shows more units than were read: carry forward units seen
        # on the board in the last few read frames that have since vanished
        # (their bar hidden behind a tooltip, a short unit's bar not found).
        target = counts.get(k)
        if target is None and FILL_TO_TEAM_SIZE and computed is not None:
            target = computed
        bench_now = Counter(u["name"] for at, u in units.items() if at[0] == "s")
        if target is not None:
            board_now = [at for at in units if at[0] == "h"]
            used = sum(slots(units[at]["name"]) for at in board_now)
            if used < target:
                window = {at: (u, k0) for at, (u, k0) in last_seen.items() if k - k0 <= CARRY_LOOKBACK}
                # Most copies of each champion on the board in any one recent
                # frame: a unit that just moved hexes is in the window twice
                # (old and new hex) but must not be carried as a second copy.
                before, bench_before = Counter(), Counter()
                for c in recent_counts[-CARRY_LOOKBACK:]:
                    before |= c
                for c in recent_bench[-CARRY_LOOKBACK:]:
                    bench_before |= c
                now = Counter(units[at]["name"] for at in board_now)
                gone = [(at, u, k0) for at, (u, k0) in window.items() if at not in units]
                for at, u, k0 in sorted(gone, key=lambda t: (-t[2], -t[1]["support"])):
                    need = slots(u["name"])
                    if need == 0 or used + need > target or now[u["name"]] >= before[u["name"]]:
                        continue
                    if counts.get(k) is None and bench_now[u["name"]] > bench_before[u["name"]]:
                        continue                     # moved to the bench, not hidden
                    units[at] = dict(u, carried=True)
                    now[u["name"]] += 1
                    used += need
                    stats["units_carried"] += 1
                    if used >= target:
                        break
        recent_counts.append(Counter(u["name"] for at, u in units.items() if at[0] == "h"))
        recent_bench.append(bench_now)
        for at, u in units.items():
            if at[0] == "h":
                last_seen[at] = (u, k)
        for at in [a for a in last_seen if a not in units and k - last_seen[a][1] > CARRY_LOOKBACK]:
            del last_seen[at]

        out[i] = {
            "board": [{"name": u["name"], "star": u["star"], "items": u["items"],
                       "hex": [int(at[1]), int(at[2])], "p": u["p"], **_flags(u)}
                      for at, u in sorted(units.items()) if at[0] == "h"],
            "bench": [{"name": u["name"], "star": u["star"], "items": u["items"],
                       "slot": int(at[1:]), "p": u["p"], **_flags(u)}
                      for at, u in sorted(units.items(), key=lambda kv: int(kv[0][1:]) if kv[0][0] == "s" else -1)
                      if at[0] == "s"],
            "read": True,
        }

    # Carry the last planning board through combat and away frames.
    last = None
    for i in range(len(rows)):
        if out[i]["read"]:
            last = out[i]
        elif last is not None:
            out[i] = {"board": last["board"], "bench": last["bench"], "read": False}
    return out, stats
