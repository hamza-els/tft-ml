# TFT ML project

Offline advisor for Teamfight Tactics: read game state out of stream VODs,
then learn what strong players do. It must never click in a live game (Riot
Vanguard would ban the account).

Owner: Hamza. He labels data and makes the design calls; explain results in
plain language and say plainly what was and wasn't verified.

## Where things are

- **Repo:** `G:\My Drive\TFT Project` (Google Drive synced), GitHub `hamza-els/tft-ml`, commits go straight to `main`.
- **Frames and VODs are NOT in the repo.** They live in `~/tft_frames_temp/<vod_id>/` and `~/tft_vods_temp/`, and get purged between sessions. Check they exist before assuming.
  - `game_00/ … game_NN/` deduped in-game frames; `lobby/` everything else; `.pipeline_steps` records finished steps.
  - Frame numbers are original: `frame_number / 2` = seconds into the VOD (2 fps).
- **Gitignored outputs:** `data/raw/`, `data/processed/state_vectors/`, `data/datasets/`, `data/processed/champion_crops/`, `data/processed/unit_crops/images/`, `data/processed/labeled/images/`.
- **In git:** code, `configs/`, `assets/templates/` (icon templates), `assets/game_data/`, `data/processed/unit_crops/{index,labels}.json`.
- `data/raw/review/` holds pictures made for Hamza to check by eye.

## Pipeline, in order

```
python scripts/run_pipeline.py                    # urls in data/raw/urls.txt: download, extract, split games, dedup
python scripts/extract_states.py --frames ~/tft_frames_temp/VOD   # OCR etc. -> state_vectors/VOD/game_XX.jsonl  (slow, ~2 h per 9 games)
python scripts/clean_states.py   --states data/processed/state_vectors/VOD   # -> game_XX.clean.jsonl + quality report
python scripts/build_dataset.py  --states data/processed/state_vectors/VOD   # -> data/datasets/states/VOD/game_XX.npz
```

Helpers:

| Script | Use |
|---|---|
| `preview_regions.py <frame>` | Draw every region on a frame. Run this first on any new footage. |
| `calibrate_regions.py <frame>` | Click to measure a region. |
| `backfill_states.py` | Recompute the non-OCR fields in existing state files (`--reread-hp` also redoes HP). Much faster than re-extracting. |
| `collect_unit_crops.py` → `label_units.py` | Cut unit crops, then Hamza labels them (Tk app, fuzzy search). |
| `prune_unit_crops.py` | Move combat/away crops out of the labeling set. |
| `harvest_augment_icons.py`, `harvest_item_icons.py` | Collect icon templates from footage. |
| `harvest_bench_units.py` | Auto-label bench crops from shop purchases (noisy; used as suggestions only). |
| `fetch_game_data.py`, `fetch_item_icons.py` | Set data and item art from Community Dragon. |

Tests: `python -m pytest tests -q` (about 40 s; the OCR tests load EasyOCR and use the two frames in `tests/fixtures/`).

## Things that will bite you

- **Everything is calibrated to one HUD layout**: BoxBox's stream, VOD `knB6dE8Rbgg`, TFT Set 18. Regions in `configs/vision_config.yaml`, `WATCH_REGIONS` in `frame_dedup.py`, and the templates in `assets/templates/hud/` are all specific to it. Another streamer or UI scale gives silent garbage or "no games detected". Run `preview_regions.py` before processing anything new.
- **A new TFT set** needs `fetch_game_data.py --set ...`, new icon templates, and changes the state vector's size (champion and trait counts).
- **The HUD is always the player's own, the board is not.** About 28% of frames show another player's board (away fights, scouting). `home_board` marks own-board frames: the augment bar on the left means own board, on the right means someone else's, arena colour is the fallback. Never read units or augments off a frame that isn't `home_board`.
- **Planning vs combat** comes from the round timer bar refilling (`planning_rows` in `src/vision/arena.py`). Units are only read in planning; Hamza decided mid-combat unit data is not needed.
- **The streamer often hides the shop bar**, so "gold icon missing" does not mean carousel. Bottom-bar values are `None` then, not guessed.
- **The stage bar shifts right during stage 1**; stage text is read relative to the detected icon.
- **HP can rise and exceed 100** (augments). Two panels can replace the player list: the "Damage Dealt" recap (own HP still shown at the top) and a unit info panel (no HP).
- **Dedup deletes frames permanently.** Split into games before dedup. Hovering a shop card opens a tooltip that covers half the bench.
- **Windows shell quirks:** set `PYTHONIOENCODING=utf-8` when piping script output (box-drawing characters crash cp1252). Long bash heredocs with apostrophes have failed; write a patch script to the scratchpad instead. Don't run two jobs that rewrite the same `game_XX.jsonl` at once.
- **yt-dlp** needs to be recent and needs `--js-runtimes node` (already in `run_pipeline.py`), or YouTube returns 403 mid-download.

## Decisions already made (don't reopen without reason)

- Champion classes are names only, no star level in the label. Star level is read separately later.
- The shop is read by OCR of card names, not by a detector.
- Board and bench units are cut by fixed position (hex and slot windows), then classified. The YOLO path (`train_yolo.py`, `champion_detector.py`, `data/processed/labeled/`) predates this and is probably unnecessary.
- Other players' boards are ignored. Scouting is not modelled.
- Augment and item icons are matched to templates cut from footage and identified by ID (`aug_007`, `item_031`). Names are optional: the model only needs consistent IDs.
- No hard-coded thresholds in the decision logic; the Win Predictor's output is to be a model input.

## State on 2026-11-16

Working, on 9 games: game splitting, dedup, stage, gold, level, XP, streak, own and lobby HP, shop, augment count and tier, unequipped items, own-board and planning flags, cleaning, 362-number encoding, action derivation.

Not working yet:

1. **Champion recognition on board and bench.** Blocked on labels. 189 of 1,483 crops labelled in `data/processed/unit_crops/labels.json`. Next step once there are about 500: train a small classifier on the crops (not written), plug it into `label_units.py` suggestions, report per-champion accuracy on held-out crops, then fill `board`/`bench` in the states on own-board planning frames and carry them through combat.
2. **Star level and items on units.** Both visible inside the unit crops (star marker beside the health bar, item icons under it). Do after 1.
3. **Hex position** falls out of which window a unit was found in; not written into the state yet.
4. **Gameplay model.** Framework in `docs/gameplay_model.md`; action labels and model shapes exist, no dataset labels or training loop.

## Known problems, unverified or unfixed

- Game 2's HP read rate is 85.7%, lowest of the nine; cause not looked at.
- `assets/templates/augments/` has 48 icons for 9 games (27 is the maximum), so it holds duplicates or junk. Augment tier has only ever read "gold"; the silver and prismatic rules are untested.
- 41 frames showed both augment bars; they are treated as away without having been inspected.
- Item matching against official art is only trusted above 0.75; 127 footage-cut item icons are unnamed and some may be cursor or overlay junk.
- `harvest_bench_units.py` crops were roughly a quarter to a third wrong when checked by eye.
- Action labels: SELL is over-counted and BUY_XP looks under-counted (details in `docs/gameplay_model.md`).
- `scripts/split_games.py` and `src/data_collection/game_splitter.py` (video-based splitting) use an older brightness heuristic and were not updated or tested with the new regions.
- `frame_sample_fps` in `vision_config.yaml` is unused; the pipeline's `--fps` (default 2.0) is what counts, and several places assume 2 fps.
- `tests/fixtures/` holds two stream screenshots showing the streamer's webcam; swap for crops if the repo is public and that matters.

## Suggestions

- Get more same-layout VODs before anything else; rare champions (5-costs) have few or no crops.
- `extract_states.py` is the slow step (EasyOCR, about 0.5 s per frame). The HP strip read is the most expensive part; reading it once per round instead of per frame would roughly halve the time.
- Store per-layout calibration under a layout name in the config, so a second streamer doesn't mean overwriting the first.
- Split multi-action transitions instead of dropping them as UNKNOWN.
- Add a per-game summary (placement, rounds played, augments) to the dataset; the placement label is needed for training anyway.
