"""Ranking metrics, matching trec_eval / BEIR conventions (linear gain, log2 discount)."""

from __future__ import annotations

import math

import numpy as np


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    relevant = {d for d, g in rel.items() if g > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if rel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """Mean metrics over all labelled queries (a query missing from `run` scores 0)."""
    out = {}
    for name, fn, k in [("ndcg@10", ndcg_at_k, 10), ("mrr@10", mrr_at_k, 10),
                        ("recall@10", recall_at_k, 10), ("recall@100", recall_at_k, 100)]:
        out[name] = float(np.mean([fn(run.get(q, []), rel, k) for q, rel in qrels.items()]))
    return out
