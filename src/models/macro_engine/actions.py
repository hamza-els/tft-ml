"""
The Macro Engine's action space, and how actions are read out of footage.

Nobody labels what the player did. It is recovered from how the state changes
between two consecutive frames of the same round:

  BUY slot i   shop slot i goes from a champion to empty, gold drops by its cost
  REROLL       gold drops by 2 and the shop is replaced
  BUY_XP       gold drops by 4 and XP rises by 4 (or the level goes up)
  SELL         gold rises mid-round (income only arrives between rounds)
  PASS         nothing above happened

A transition that fits none of these cleanly (several actions between two
frames, a misread, the bottom bar hidden) is UNKNOWN and is left out of
training rather than guessed.

See docs/gameplay_model.md for how these labels are used.
"""

from dataclasses import dataclass
from enum import IntEnum

from src.state.game_data import GameData

REROLL_COST = 2
XP_COST = 4
XP_GAIN = 4
SHOP_SLOTS = 5


class ActionType(IntEnum):
    PASS = 0
    BUY = 1        # arg = shop slot 0-4
    REROLL = 2
    BUY_XP = 3
    SELL = 4       # which unit is unknown until board/bench are read
    UNKNOWN = 5    # not used as a training target


# Flat list the policy head predicts over: PASS, BUY 0-4, REROLL, BUY_XP, SELL.
POLICY_ACTIONS: list[tuple[ActionType, int | None]] = (
    [(ActionType.PASS, None)]
    + [(ActionType.BUY, i) for i in range(SHOP_SLOTS)]
    + [(ActionType.REROLL, None), (ActionType.BUY_XP, None), (ActionType.SELL, None)]
)


@dataclass(frozen=True)
class Action:
    type: ActionType
    arg: int | None = None        # shop slot for BUY
    champion: str | None = None   # what was bought, for BUY

    @property
    def policy_index(self) -> int | None:
        """Index into POLICY_ACTIONS, or None for UNKNOWN."""
        key = (self.type, self.arg if self.type is ActionType.BUY else None)
        return POLICY_ACTIONS.index(key) if key in POLICY_ACTIONS else None


def derive_action(before: dict, after: dict, game: GameData) -> Action:
    """
    The action that turned `before` into `after` (two consecutive cleaned
    state rows). Returns UNKNOWN when the change can't be explained by exactly
    one action.
    """
    unknown = Action(ActionType.UNKNOWN)
    if not (before.get("shop_visible") and after.get("shop_visible")):
        return unknown
    if (before.get("stage"), before.get("round_num")) != (after.get("stage"), after.get("round_num")):
        return unknown   # income, interest and the free reroll land between rounds
    g0, g1 = before.get("gold"), after.get("gold")
    shop0, shop1 = before.get("shop_slots") or [], after.get("shop_slots") or []
    if g0 is None or g1 is None or len(shop0) != SHOP_SLOTS or len(shop1) != SHOP_SLOTS:
        return unknown

    spent = g0 - g1
    changed = [i for i in range(SHOP_SLOTS) if shop0[i] != shop1[i]]

    if spent == 0 and not changed:
        return Action(ActionType.PASS)

    # One slot emptied and gold fell by that champion's cost.
    if len(changed) == 1 and shop0[changed[0]] and shop1[changed[0]] is None:
        name = shop0[changed[0]]
        if game.cost(name) and spent == game.cost(name):
            return Action(ActionType.BUY, changed[0], name)
        return unknown

    xp0, xp1 = before.get("xp_current"), after.get("xp_current")
    levelled = (before.get("level") is not None and after.get("level") is not None
                and after["level"] > before["level"])
    if spent == XP_COST and not changed:
        if levelled or (xp0 is not None and xp1 is not None and xp1 - xp0 == XP_GAIN):
            return Action(ActionType.BUY_XP)
        return unknown

    # A reroll replaces the whole shop; a couple of slots can repeat by chance.
    if spent == REROLL_COST and len(changed) >= 3 and not levelled:
        return Action(ActionType.REROLL)

    if spent < 0 and not changed:
        return Action(ActionType.SELL)

    return unknown


def derive_actions(rows: list[dict], game: GameData) -> list[Action]:
    """Action for each consecutive pair of rows: len(rows) - 1 entries."""
    return [derive_action(a, b, game) for a, b in zip(rows, rows[1:])]
