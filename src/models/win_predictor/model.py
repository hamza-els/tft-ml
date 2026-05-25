"""
Siamese MLP win predictor.
Input: two board state embeddings (your board + opponent's last-known board).
Output: P(win) scalar in [0, 1].

Stub — architecture wired up, training loop TBD.
"""

import torch
import torch.nn as nn


class BoardEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WinPredictor(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.encoder = BoardEncoder(input_dim, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, board_a: torch.Tensor, board_b: torch.Tensor) -> torch.Tensor:
        emb_a = self.encoder(board_a)
        emb_b = self.encoder(board_b)
        combined = torch.cat([emb_a, emb_b], dim=-1)
        return self.head(combined).squeeze(-1)
