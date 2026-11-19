we still need to clean up some frames where there is combat with bots on our own board

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
- **All pipeline work happens on the local disk, never in the Drive folder** (Hamza's rule, 2026-11-18, after Drive locked a file mid-rename and later dropped out mid-run). Paths in `src/utils/paths.py`, root `~/tft_data` (env `TFT_DATA`):
  - `~/tft_data/state_vectors/<vod>/` working copy of every state file (`.done` marks a finished VOD for `process_vods.py`)
  - `~/tft_data/unit_crops/images/`, `.../removed/` crop images; `~/tft_data/datasets/states/` encoded `.npz`; `~/tft_data/logs/process_vods.log`
  - Only finished state files are copied into the repo's `data/processed/state_vectors/` (`scripts/upload_states.py`, the last step of `process_vods.py`). Small repo files still written by the pipeline: `unit_crops/index.json`, `labels.json`, icon templates.
- **Gitignored outputs:** `data/raw/`, `data/processed/state_vectors/`, `data/datasets/`, `data/processed/champion_crops/`, `data/processed/labeled/images/`.
- **In git:** code, `configs/`, `assets/templates/` (icon templates), `assets/game_data/`, `data/processed/unit_crops/{index,labels}.json`, `.../removed/labels.json`.
- `data/raw/review/` holds pictures made for Hamza to check by eye.

## Pipeline, in order

```
python scripts/process_vods.py --urls FILE        # all of the below, one VOD at a time (preferred)

python scripts/run_pipeline.py                    # urls in data/raw/urls.txt: download, extract, split games, dedup
python scripts/extract_states.py --frames ~/tft_frames_temp/VOD   # OCR etc. -> ~/tft_data/state_vectors/VOD/game_XX.jsonl
python scripts/read_units.py     --frames ~/tft_frames_temp/VOD --states ~/tft_data/state_vectors/VOD   # board/bench units -> units_raw (re-run after retraining)
python scripts/clean_states.py   --states ~/tft_data/state_vectors/VOD   # -> game_XX.clean.jsonl (incl. cleaned board/bench) + quality report
python scripts/build_dataset.py  --states ~/tft_data/state_vectors/VOD   # -> ~/tft_data/datasets/states/VOD/game_XX.npz
python scripts/upload_states.py VOD               # copy the finished state files into the repo
```

Helpers:

| Script | Use |
|---|---|
| `preview_regions.py <frame>` | Draw every region on a frame. Run this first on any new footage. |
| `calibrate_regions.py <frame>` | Click to measure a region. |
| `backfill_states.py` | Recompute the non-OCR fields in existing state files (`--reread-hp` also redoes HP). Much faster than re-extracting. |
| `collect_unit_crops.py` → `label_units.py` | Cut unit crops, then Hamza labels them (Tk app, fuzzy search). `--unsure` queues only crops the classifier is below 90% sure of, least sure first (most value per label; scores cached in `confidence_cache.json` per model). `--relabel LABEL` goes through every crop with that label again. Tab = `_none` (not a unit), Shift+Tab = `_unknown` (a unit, can't tell which). Saves merge into labels.json, so other scripts can change it while the labeler is open. |
| `prune_unit_crops.py` | Move combat/away crops (incl. combat start: timer empty or "Combat" banner) out of the labeling set; labels kept in `removed/labels.json`. |
| `recut_unit_crops.py` | Re-cut every crop from its frame after changing the window settings (labels survive; retrain after). |
| `read_units.py` | Classify board/bench units, star level and items on own-board planning frames -> `units_raw`, plus the board counter -> `board_counter`. Re-run after retraining; `--counter-only` re-reads just the counter. |
| `build_counter_templates.py` | Rebuild `assets/templates/board_counter/` from cleaned states and frames. |
| `upload_states.py` | Copy finished state files from `~/tft_data` into the repo. |
| `harvest_unit_item_icons.py` | Rebuild `assets/templates/unit_items/` (icons under units' health bars) from the crops' frames. |
| `harvest_augment_icons.py`, `harvest_item_icons.py` | Collect icon templates from footage. |
| `harvest_bench_units.py` | Auto-label bench crops from shop purchases (noisy; used as suggestions only). |
| `fetch_game_data.py`, `fetch_item_icons.py` | Set data and item art from Community Dragon. |

Tests: `python -m pytest tests -q` (about 40 s; the OCR tests load EasyOCR and use the two frames in `tests/fixtures/`).

## Things that will bite you

- **Everything is calibrated to one HUD layout**: BoxBox's stream, VOD `knB6dE8Rbgg`, TFT Set 18. Regions in `configs/vision_config.yaml`, `WATCH_REGIONS` in `frame_dedup.py`, and the templates in `assets/templates/hud/` are all specific to it. Another streamer or UI scale gives silent garbage or "no games detected". Run `preview_regions.py` before processing anything new.
- **A new TFT set** needs `fetch_game_data.py --set ...`, new icon templates, and changes the state vector's size (champion and trait counts).
- **The HUD is always the player's own, the board is not.** About 28% of frames show another player's board (away fights, scouting). `home_board` marks own-board frames: the augment bar on the left means own board, on the right means someone else's, arena colour is the fallback. Never read units or augments off a frame that isn't `home_board`.
- **Planning vs combat** comes from the round timer bar refilling (`planning_rows` in `src/vision/arena.py`), plus: no enemy bars on the far half, timer not empty, and no "Combat" banner under the stage bar (`combat_banner`, greyscale template of the word; a "Planning" banner sits at the same spot and scores 0.5-0.6 against it, "Combat" 0.6-1.0). Frames at combat start show units turned to face the enemy, which confused the classifier. Units are only read in planning; Hamza decided mid-combat unit data is not needed.
- **The streamer often hides the shop bar**, so "gold icon missing" does not mean carousel. Bottom-bar values are `None` then, not guessed.
- **The stage bar shifts right during stage 1**; stage text is read relative to the detected icon.
- **HP can rise and exceed 100** (augments). Two panels can replace the player list: the "Damage Dealt" recap (own HP still shown at the top) and a unit info panel (no HP).
- **State reading skips OCR of unchanged regions** (`OcrCache` in `ocr_pipeline.py`): a region is re-read only when its bright text pixels changed, and a value is reused only after two reads agreed. HP is read once per round (until two own-HP reads agree and the lobby list was read): the HP column changes on almost every frame (notices sliding over names, animated portrait rings), so it can't be change-gated. Consequence: between a fight ending and the next round's counter, `hp` still shows the pre-fight value; lobby HP is the round-start value. Checked on 300 frames: 66% of OCR skipped, stage/level/XP/streak/shop identical to per-frame reading, gold differed on 2 frames (an OCR misread both versions share).
- **Google Drive locks files while syncing**: a rename in `extract_states.py` failed with WinError 32 and lost a VOD's run; it now retries. Other writers may need the same if it recurs.
- **Dedup deletes frames permanently.** Split into games before dedup. Hovering a shop card opens a tooltip that covers half the bench.
- **Windows shell quirks:** set `PYTHONIOENCODING=utf-8` when piping script output (box-drawing characters crash cp1252). Long bash heredocs with apostrophes have failed; write a patch script to the scratchpad instead. Don't run two jobs that rewrite the same `game_XX.jsonl` at once.
- **yt-dlp** needs to be recent and needs `--js-runtimes node` (already in `run_pipeline.py`), or YouTube returns 403 mid-download.

## Decisions already made (don't reopen without reason)

- Champion classes are names only, no star level in the label. Star level is read separately later.
- The shop is read by OCR of card names, not by a detector.
- Board and bench units are found by their health bar (`unit_reader.occupied`: a green bar lined up with the hex/slot with a star badge on its left; the badge rules out the Little Legend's nameplate bar), cut with a window anchored to that bar (`board.anchored`: fixed size, centred on the bar, top just above the badge), then classified. Hamza chose anchored windows over fixed ones (2026-11-18): held-out accuracy went from 91.5% to 93.7%. The YOLO path (`train_yolo.py`, `champion_detector.py`, `data/processed/labeled/`) predates this and is probably unnecessary.
- Other players' boards are ignored. Scouting is not modelled.
- Augment and item icons are matched to templates cut from footage and identified by ID (`aug_007`, `item_031`). Names are optional: the model only needs consistent IDs.
- No hard-coded thresholds in the decision logic; the Win Predictor's output is to be a model input.

## State on 2026-11-18

Working: game splitting, dedup, stage, gold, level, XP, streak, own and lobby HP, shop, augment count and tier, unequipped items, own-board and planning flags, cleaning, 376-number encoding, action derivation, and now board/bench units with star level, items and hex/slot (in the cleaned states).

1. **Units in the states.** `read_units.py` writes single-frame readings (`units_raw`); `clean_states.py` turns them into `board` / `bench` per frame (`src/state/units.py`): each frame weighted by how long it stood unchanged (dedup removed its copies; capped at 10 s) times classifier confidence; per hex/slot, a flicker of up to 2 frames between two agreeing longer runs takes their value (empty readings count half: a hidden bar is the commonest flicker); star and items smoothed within each stretch of one unit; board capped at the team size: level + bonuses from the set's own data (`fetch_game_data.py` now stores them: Riftbeast 10 +2 team size, Elder Dragon takes 2 slots and +2 Riftbeast, Tactician's Crown/Cape/Shield and Choncc's items +1/+2 while held, 20 emblems that add their trait); trait/augment units (`extra_units`: Elderwood plants, BFF, ...) stand on the board but take no slot, as in the game's own "units / team size" counter. Augment team-size bonuses (New Recruit +1, Cursed Crown +2, Endless Hordes +3) are invisible in the rules (augment IDs are unnamed), so the game's own counter comes first: `src/vision/board_counter.py` reads the big translucent "units / team size" in the arena (edge templates per number and side, built by `scripts/build_counter_templates.py`; the slash stays put, "10" reaches out on both sides). On 600 planning frames: team size read on 76%, equal to the level on 98.9% of those; the unit count is read less often (units, nameplates, PvE monsters' bars cover it) and one frame alone can be wrong, so it must agree with nearby frames. Team size per round = most common reading (2+ agreeing, between level and level + 3); the rules only fill in where it isn't read. The counter often shows one unit more than we read (a missed unit, e.g. behind a tooltip): counted in the quality report as "counter shows more units than read"; last planning board carried through combat. On knB6dE8Rbgg: about 1,900 one-frame dropouts filled, 70 misreads fixed, over the level on 14 of ~3,400 planning frames. Over-level cases looked at: a unit being dragged, and crowded units where one bar seems claimed by two hexes.
2. **Classifier** (`src/vision/unit_classifier.py`, ResNet-18, `train_unit_classifier.py`, weights gitignored). 2026-11-18, 4,204 labels, anchored crops: 93.7% right first guess on held-out knB6dE8Rbgg game_02, 98.1% in the top 3. Misses: champions with almost no labels (Tristana 3, Rammus 4, Ashe 2, Big Furry Friend 0). Labels `_none` (not a unit) and `_unknown` (unit, can't tell) are both classes; `_unknown` counts as occupied.
3. **Stars and items** (`src/vision/unit_bar.py`): stars read on 98% of units, right on all ~120 checked by eye. Items: count right on 18/21 checked; names less certain (Radiant look-alikes, some items with two IDs).
4. **Not encoded yet:** hex positions and items on units are in the cleaned rows but not in the 376-number vector.
5. **Gameplay model.** Framework in `docs/gameplay_model.md`; action labels and model shapes exist, no dataset labels or training loop.

## Known problems, unverified or unfixed

- knB6dE8Rbgg game 2's HP read rate is 86%, the lowest of all games; cause not looked at. (It is not the dark-arena problem fixed on 2026-11-18: the panel check used to judge "panel open" by darkness, so dark arenas hid the player list; it now matches the panel's exact teal colour, which lifted eUOJYk4MKaY games 7-9 from 60-82% to 93-98%.)
- YouTube's "confirm you're not a bot" block tends to return after about 3 VOD downloads in a row. `process_vods.py` stops the batch when it sees it; wait a few hours and re-run (finished VODs are skipped).
- `assets/templates/augments/` has 48 icons for 9 games (27 is the maximum), so it holds duplicates or junk. Augment tier has only ever read "gold"; the silver and prismatic rules are untested.
- 41 frames showed both augment bars; they are treated as away without having been inspected.
- Item matching against official art is only trusted above 0.75. Footage-cut item templates were rebuilt on 2026-11-17 after fixing the slot spacing (54 px, not 53): 69 icons, 51 named automatically, 18 unnamed.
- `data/raw/urls.txt` holds 10 more BoxBox VODs (picked 2026-11-17 by sampling frames for the calibrated HUD). VR23D04Dte0 was streamed from a hotel room with a different overlay; check `preview_regions.py` on its frames before trusting it. Bash tournament days were skipped because parts show a spectator/broadcast view, not his own game.
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
