# Gameplay model: framework

Status: design plus a code skeleton. Nothing is trained. Numbers below are
from the one processed VOD (9 games, 10,686 states) on 2026-11-16.

## What it is for

An offline advisor: given what is on screen now and what led up to it, say
what a strong player would do next (buy, roll, level, hold) and how well the
game is going. It never clicks anything (Riot Vanguard would ban the account).

## Pieces and how they connect

```
frames ──> state rows ──> cleaned rows ──> encoded states (376 numbers per frame)
                               │                    │
                               │                    ▼
                               └──> actions ──> Macro Engine ──> policy   (what to do next)
                                    (derived)        ▲           placement (how the game ends)
                                                     │           hp_next   (HP lost this round)
                              Win Predictor ─────────┤
                              Target Comp ───────────┘   (both feed in as extra input features)
```

| Piece | File | State |
|---|---|---|
| State encoder | `src/state/encoder.py` | Working. Board and bench blocks are zeros until units are recognised. |
| Action labels | `src/models/macro_engine/actions.py` | Working, tested. |
| Macro Engine | `src/models/macro_engine/model.py` | Shapes and heads fixed, no training loop. |
| Win Predictor | `src/models/win_predictor/model.py` | Old stub. Needs boards, so blocked on unit recognition. |
| Target Comp | `src/models/target_comp/recommender.py` | Old stub. Needs the meta scraper output. |

## Inputs

One encoded state per kept frame, in order, for one game. `StateEncoder.describe()`
prints the layout. Frames are irregularly spaced (dedup keeps a frame on change
or every 5 s), so `time_in_game` and `round_index` are part of the vector
rather than assumed from position.

The masks at the end of the vector (`bottom_bar_visible`, `has_board`,
`on_home_board`, `planning`) tell the model which blocks to trust.

Next to `X`, each game's `.npz` holds a per-unit token table (`src/state/tokens.py`,
decided with Hamza 2026-11-19): 38 rows per frame (board 14, bench 9, shop 5,
unequipped items 10), padded, with `tok_mask`. Integer fields (`tok_int`): group,
champion ID, star, three item IDs with radiant flags (Radiant = base item + flag),
item ID for item-column rows, hex row/col, slot. Float fields (`tok_float`):
reading certainty (confidence, carried, filled, loose), cost, and the champion's
trait flags (own traits plus emblems) as fixed inputs, so rare champions' traits
are known without having to learn them. IDs are fixed by the set data
(`vocab.json` next to the datasets). `placement` (the final-placement label) is
saved per game. Gold, HP, round, level, XP stay in `X`.

## Action space

Nine choices (`POLICY_ACTIONS`): PASS, BUY slot 0–4, REROLL, BUY_XP, SELL.

Actions are not hand-labelled. `derive_action` reads them from how two
consecutive rows of the same round differ (gold, shop slots, level, XP). A
transition that can't be explained by exactly one action is UNKNOWN and is
masked out of the policy loss.

Measured on the 9 games:

| Action | Count |
|---|---|
| PASS | 5,525 |
| UNKNOWN | 4,217 (39%) |
| SELL | 427 |
| BUY | 268 |
| REROLL | 218 |
| BUY_XP | 22 |

Known weaknesses, to fix before trusting these:

- **SELL is over-counted.** Any mid-round gold rise is called a sell, but gold
  also rises from orbs and some augments. Once the bench is read, require a
  unit to disappear.
- **BUY_XP looks under-counted** (22 in 9 games). The rule assumes 4 gold for
  4 XP; check that against this set, and check whether XP purchases are being
  lost to UNKNOWN because two presses land between frames.
- **39% UNKNOWN** is mostly round boundaries and hidden-bar frames (correct to
  skip) but also fast multi-action bursts. Bursts could be split into their
  parts (bought two units and rolled once) instead of dropped.
- **SELL and positioning carry no argument** (which unit, which hex) until the
  board and bench are read.
- **PASS dominates.** Weight the loss by class, or only score PASS on frames
  that end a planning phase ("chose to do nothing more this round").

## Model

`MacroEngine`: a causal Transformer over the state sequence. Each position
sees only earlier frames and outputs three things:

- **policy** — the next action. This is the advisor.
- **placement** — final placement 1–8 for the game. Label: one more than the
  number of other players still alive at the player's last frame (`other_hp`).
- **hp_next** — HP lost in the current round's fight. Label: HP at the start
  of the round minus HP at the start of the next.

Placement and hp_next exist for every frame at no labelling cost. They teach
the model what a strong position looks like, which is what makes the policy
head learnable from few action examples.

Defaults (`configs/model_config.yaml`): 256 wide, 4 heads, 4 layers. That is
too big for 9 games; start at 64 wide, 2 layers and grow with the data.

## Training plan

1. **Dataset.** Extend `scripts/build_dataset.py` to also save, per game:
   action index per transition (−1 for UNKNOWN), placement, hp_next. Split
   train/validation by whole game, and by whole VOD once there are several.
2. **Baselines first.** Before the Transformer, score (a) always-PASS and
   (b) a small MLP on the current state only. The sequence model has to beat
   both to justify itself.
3. **Stage 1: outcome heads only.** Train placement and hp_next. Works with no
   action labels at all and shows whether the state vector carries signal.
4. **Stage 2: add the policy head.** Behaviour cloning with a masked,
   class-weighted cross-entropy.
5. **Stage 3: add the board.** Board and bench are now read (units, stars,
   items, hexes) and in both `X` and the token table; SELL can get an argument.
6. **Stage 4: side models.** Win Predictor output and Target Comp distance
   become extra input features. No hard-coded thresholds (project decision).

## How to judge it

- Policy: accuracy and per-class recall on non-PASS actions, on held-out games.
  Overall accuracy is meaningless because PASS dominates.
- Placement: mean absolute error in places; compare with always guessing 4.5.
- hp_next: mean absolute error; compare with predicting the stage average.
- Sanity checks on behaviour: does it roll more at low HP, level at the usual
  stage breakpoints, hold gold for interest above 50?

## Data needed

Nine games is a smoke test, not a training set. Behaviour cloning of this kind
usually needs hundreds of games before the policy is better than simple rules.
Same-layout VODs (BoxBox) go through the pipeline unchanged; another streamer
needs the regions recalibrated first (see `CLAUDE.md`).

All footage so far is one very strong player. The model will learn that
player's style, and "what this player does" is being used as "what is good".
Mixing in several strong players is the fix.

## Open decisions for Hamza

- Which actions the advisor should name. The nine above cover economy; is
  positioning in scope?
- Advice per frame, or once per round ("this round: roll to 30, then level")?
  Per-round is easier to learn and easier to read.
- Whether to keep opponents out entirely. Hamza said (2026-11-16) to ignore
  other players' boards; lobby HP is still in the vector.
