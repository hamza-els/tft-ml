"""
Tracks the last known board state of each opponent in the lobby.
Applies a staleness counter so the Macro Engine can weight fresh vs. stale intel.
"""

from dataclasses import dataclass, field

from src.state.state_vector import UnitState


@dataclass
class OpponentSnapshot:
    opponent_id: str
    board: list[UnitState]
    hp: float | None
    level: float | None
    last_seen_stage: float | None
    staleness: int = 0  # incremented each round without a fresh scout


class OpponentMemory:
    def __init__(self, num_players: int = 7):
        self._memory: dict[str, OpponentSnapshot] = {}
        self._num_players = num_players

    def update(self, opponent_id: str, snapshot: OpponentSnapshot) -> None:
        snapshot.staleness = 0
        self._memory[opponent_id] = snapshot

    def age_all(self) -> None:
        """Call once per round to increment staleness on all unscouted opponents."""
        for snap in self._memory.values():
            snap.staleness += 1

    def get(self, opponent_id: str) -> OpponentSnapshot | None:
        return self._memory.get(opponent_id)

    def all_snapshots(self) -> list[OpponentSnapshot]:
        return list(self._memory.values())

    def staleness_vector(self) -> dict[str, int]:
        """Returns {opponent_id: staleness} for all tracked opponents."""
        return {k: v.staleness for k, v in self._memory.items()}
