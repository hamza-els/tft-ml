"""
Macro Engine — Transformer over a game's sequence of encoded states.

Input:  a window of consecutive state vectors (src/state/encoder.py).
Output: three heads, read at every position (each position only sees the past):
  policy     logits over POLICY_ACTIONS — what the player did next
  placement  logits over final placement 1-8 — how the game ended up
  hp_next    predicted HP lost in the current round's fight

The policy head is the advisor. The other two are auxiliary targets that are
available for every frame for free and force the model to learn what a strong
position looks like, which matters when action labels are scarce.

Framework only: shapes and heads are fixed here, the training loop is not
written yet. See docs/gameplay_model.md.
"""

import torch
import torch.nn as nn

from src.models.macro_engine.actions import POLICY_ACTIONS

NUM_PLACEMENTS = 8


class MacroEngine(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.position = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.policy_head = nn.Linear(hidden_dim, len(POLICY_ACTIONS))
        self.placement_head = nn.Linear(hidden_dim, NUM_PLACEMENTS)
        self.hp_head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor, padding_mask: torch.Tensor | None = None) -> dict:
        """
        Args:
            states: (batch, seq_len, state_dim) consecutive state vectors, oldest first.
            padding_mask: (batch, seq_len) True where a position is padding.

        Returns:
            dict of per-position outputs:
              policy    (batch, seq_len, len(POLICY_ACTIONS))
              placement (batch, seq_len, 8)
              hp_next   (batch, seq_len)
        """
        seq_len = states.shape[1]
        x = self.input_proj(states) + self.position(torch.arange(seq_len, device=states.device))
        # Causal mask: a position may not look at later frames.
        causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=states.device), 1)
        x = self.transformer(x, mask=causal, src_key_padding_mask=padding_mask)
        return {
            "policy": self.policy_head(x),
            "placement": self.placement_head(x),
            "hp_next": self.hp_head(x).squeeze(-1),
        }
