"""The real model wrappers, with sentence-transformers replaced by a stub (no download)."""

import numpy as np
import sentence_transformers
import torch

from biomedrag import models


class StubCrossEncoder:
    """Returns large logits through whatever activation it is given, like the real one."""

    def __init__(self, name, device=None, max_length=None):
        self.model = torch.nn.Linear(1, 1)

    def predict(self, pairs, batch_size=64, show_progress_bar=False, activation_fn=None):
        logits = torch.tensor([13.9, 8.7, 7.7, 0.9], dtype=torch.float16)[: len(pairs)]
        act = activation_fn if activation_fn is not None else torch.nn.Sigmoid()
        return act(logits).numpy()


def test_cross_encoder_scores_are_untied_logits_not_saturated_probabilities(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", StubCrossEncoder)
    monkeypatch.setattr(models, "device", lambda: "cpu")
    pairs = [("q", "a"), ("q", "b"), ("q", "c"), ("q", "d")]
    # what the default sigmoid does in fp16: the top two tie at exactly 1.0
    sig = StubCrossEncoder("x").predict(pairs)
    assert sig[0] == sig[1] == 1.0
    scores = models._CrossEncoder("x").predict(pairs)
    assert scores.dtype == np.float32
    assert len(set(scores.tolist())) == 4 and list(np.argsort(-scores)) == [0, 1, 2, 3]
