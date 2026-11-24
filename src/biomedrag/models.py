"""The single place where neural models are loaded (tests replace these two functions)."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray: ...


class PairScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 64) -> np.ndarray: ...


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class _STEncoder:
    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device())

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                 show_progress_bar=False)


class _CrossEncoder:
    def __init__(self, name: str, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(name, device=device(), max_length=max_length)
        if device() == "cuda":
            # fp16 halves the cost of the (dominant) re-ranking stage on GPU; the README compares
            # fp16 with the fp32 run to show ranking quality is unchanged.
            self.model.model.half()

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 64) -> np.ndarray:
        """Raw logits, not sigmoid probabilities. Sigmoid preserves order but saturates: in fp16,
        logits above ~8 all become 1.0. On SciFact dev that tied scores in 52 of 100 queries
        (6 at rank 1), and the ties silently fell back to dense order."""
        import torch

        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False,
                                    activation_fn=torch.nn.Identity())
        return np.asarray(scores, dtype=np.float32)


@lru_cache(maxsize=4)
def load_encoder(name: str) -> Encoder:
    return _STEncoder(name)


@lru_cache(maxsize=4)
def load_cross_encoder(name: str) -> PairScorer:
    return _CrossEncoder(name)
