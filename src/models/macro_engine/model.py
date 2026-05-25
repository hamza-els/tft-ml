"""
Macro Engine — Transformer sequence model.
Trained on sequences of StateVectors extracted from VODs.
Predicts the next best macro action: [level_up, roll, hold, buy_unit_N, sell_unit_N].

Stub — architecture defined, training pipeline TBD.
"""

import torch
import torch.nn as nn


class MacroEngine(nn.Module):
    ACTION_SPACE = ["level_up", "roll", "hold", "buy_0", "buy_1", "buy_2", "buy_3", "buy_4"]

    def __init__(self, state_dim: int, hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(hidden_dim, len(self.ACTION_SPACE))

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_seq: (batch, seq_len, state_dim) tensor of consecutive state vectors.

        Returns:
            (batch, num_actions) logits over the macro action space.
        """
        x = self.input_proj(state_seq)
        x = self.transformer(x)
        return self.action_head(x[:, -1, :])  # predict from final token
