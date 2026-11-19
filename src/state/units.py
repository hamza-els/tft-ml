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
   lasted a while stays.

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


def frame_weights(frames: list[int], max_seconds: float = MAX_STABLE_SECONDS) -> list[float]:
    """Seconds each kept frame stood for (until the next kept frame), capped."""
    out = []
    for i, f in enumerate(frames):
        gap = (frames[i + 1] - f) / FPS if i + 1 < len(frames) else 1 / FPS
        out.append(min(max(gap, 1 / FPS), max_seconds))
    return out


def smooth(labels: list, weights: list[float], max_blip: int = MAX_BLIP_FRAMES) -> list:
    """
    Overwrite short runs that interrupt two longer runs of the same label.

    labels: one label per reading (any hashable; None is a label too, e.g. "empty").
    weights: how much each reading counts. A run is overwritten when it has at
    most `max_blip` readings and no more total weight than each neighbour run.
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
                    and blip[2] <= min(before[2], after[2])):
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


def clean_units(rows: list[dict], levels: list, game=None, rounds: list | None = None) -> tuple[list[dict], dict]:
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
             "loose_kept": 0, "loose_dropped": 0}
    rounds = rounds or [None] * len(rows)
    out = [{"board": [], "bench": [], "read": False} for _ in rows]
    if not read:
        return out, stats

    w = [weights_all[i] for i in read]
    counts = _counter_units(rows, read, rounds)
    raw = [{u["at"]: u for u in _keep_loose(rows[i]["units_raw"], counts.get(k), stats)}
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
        smoothed = smooth(names, conf_known)
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
                              "support": conf[k]}

    slots = game.slots if game is not None else (lambda name: 1)
    team_sizes = _counter_team_sizes(rows, read, levels, rounds)
    counts = _counter_units(rows, read, rounds)
    last_seen: dict[str, tuple[dict, int]] = {}      # board hex -> (unit, read-frame index last seen)
    recent_counts: list[Counter] = []                 # champion copies on the board, per read frame
    for k, i in enumerate(read):
        units = cleaned[k]
        board = sorted((at for at in units if at[0] == "h"), key=lambda at: -units[at]["support"])
        level = levels[i]
        cap = None
        if team_sizes.get(k) is not None:
            cap = team_sizes[k]
            stats["team_size_from_counter"] += 1
        elif level:
            bonus = game.team_size_bonus([units[at] for at in board]) if game is not None else 0
            cap = int(level) + bonus
            stats["team_size_bonus_frames"] += bonus > 0
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
        if counts.get(k) is not None:
            board_now = [at for at in units if at[0] == "h"]
            used = sum(slots(units[at]["name"]) for at in board_now)
            if used < counts[k]:
                window = {at: (u, k0) for at, (u, k0) in last_seen.items() if k - k0 <= CARRY_LOOKBACK}
                # Most copies of each champion on the board in any one recent
                # frame: a unit that just moved hexes is in the window twice
                # (old and new hex) but must not be carried as a second copy.
                before = Counter()
                for c in recent_counts[-CARRY_LOOKBACK:]:
                    before |= c
                now = Counter(units[at]["name"] for at in board_now)
                gone = [(at, u, k0) for at, (u, k0) in window.items() if at not in units]
                for at, u, k0 in sorted(gone, key=lambda t: (-t[2], -t[1]["support"])):
                    need = slots(u["name"])
                    if need == 0 or used + need > counts[k] or now[u["name"]] >= before[u["name"]]:
                        continue
                    units[at] = dict(u, carried=True)
                    now[u["name"]] += 1
                    used += need
                    stats["units_carried"] += 1
                    if used >= counts[k]:
                        break
        recent_counts.append(Counter(u["name"] for at, u in units.items() if at[0] == "h"))
        for at, u in units.items():
            if at[0] == "h":
                last_seen[at] = (u, k)
        for at in [a for a in last_seen if a not in units and k - last_seen[a][1] > CARRY_LOOKBACK]:
            del last_seen[at]

        out[i] = {
            "board": [{"name": u["name"], "star": u["star"], "items": u["items"],
                       "hex": [int(at[1]), int(at[2])], "p": u["p"], **({"carried": True} if u.get("carried") else {})}
                      for at, u in sorted(units.items()) if at[0] == "h"],
            "bench": [{"name": u["name"], "star": u["star"], "items": u["items"],
                       "slot": int(at[1:]), "p": u["p"]}
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
