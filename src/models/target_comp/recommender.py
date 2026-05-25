"""
Target comp recommender using cosine similarity.
Matches your current item/unit vector against the scraped meta comp database
to suggest the best endgame composition to pivot toward.

Stub — vectorization logic TBD once meta_scraper output format is finalized.
"""

import json
from pathlib import Path

import numpy as np


class TargetCompRecommender:
    def __init__(self, meta_comps_path: str):
        with open(meta_comps_path) as f:
            self.comps = json.load(f)
        # TODO: build embedding matrix once unit/item vocabulary is defined
        self._embeddings: np.ndarray | None = None

    def recommend(self, current_vector: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Return the top_k meta comps most similar to the current board state.

        Args:
            current_vector: Numerical representation of your current items/units.
            top_k: Number of recommendations to return.

        Returns:
            List of comp dicts from the scraped database, sorted by similarity.
        """
        if self._embeddings is None:
            raise RuntimeError("Call build_embeddings() before recommend().")
        sims = _cosine_similarity(current_vector, self._embeddings)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [
            {**self.comps[i], "similarity": float(sims[i])}
            for i in top_indices
        ]

    def build_embeddings(self, vocab: dict[str, int]) -> None:
        """Convert scraped comp dicts to dense vectors using the provided vocabulary."""
        # TODO: implement once unit/item vocab is locked in
        pass


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return b_norm @ a_norm
