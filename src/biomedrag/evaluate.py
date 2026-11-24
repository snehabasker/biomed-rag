"""Measure dense retrieval with and without cross-encoder re-ranking on BEIR SciFact.

* Stage 1 retrieves the top 100 candidates per query once. Each cross-encoder scores those
  100 pairs once, and every `rerank_k` in {10, 20, 50, 100} reuses the same scores.
* For each re-ranker, `rerank_k` is chosen on the **train (dev)** queries: the best nDCG@10
  among the k whose p95 latency fits the budget, preferring the smaller k when within
  0.002. If no k fits, the smallest k is kept and flagged `within_budget: false`, and the
  promotion gate will reject it. Only **test** numbers are reported as results.
* Latency is measured per query (dense search + re-ranking), on dev queries, on whatever
  device runs the evaluation (the README says which). A first fp32 run chose k by quality
  alone; its most accurate model was 9x over budget and the gate rejected it, which is why
  latency now constrains the choice of k.
"""

from __future__ import annotations

import time

import numpy as np

from biomedrag import data
from biomedrag.metrics import evaluate_run
from biomedrag.retrieve import DenseRetriever, Reranker, SearchConfig

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKERS = ("cross-encoder/ms-marco-MiniLM-L6-v2", "ncbi/MedCPT-Cross-Encoder")
RERANK_KS = (10, 20, 50, 100)
CANDIDATES = 100
TIE_TOLERANCE = 0.002
LATENCY_BUDGET_MS = 250.0  # p95 per query; the registry's promotion gate uses the same default


def choose_k(dev_ndcg: dict[int, float], tol: float = TIE_TOLERANCE, p95_ms: dict[int, float] | None = None,
             budget_ms: float = LATENCY_BUDGET_MS) -> tuple[int, bool]:
    """-> (k, within_budget): the best dev nDCG@10 among the k that fit the latency budget."""
    allowed = {k: v for k, v in dev_ndcg.items() if p95_ms is None or p95_ms[k] <= budget_ms}
    if not allowed:
        return min(dev_ndcg), False
    best = max(allowed.values())
    return min(k for k, v in allowed.items() if v >= best - tol), True


def _ids(hits) -> list[str]:
    return [h.doc.id for h in hits]


def _score_split(retriever: DenseRetriever, ds: data.Dataset, rerankers: dict[str, Reranker],
                 ks: tuple[int, ...]) -> dict[str, dict]:
    qids = list(ds.queries)
    texts = [ds.queries[q] for q in qids]
    hits = retriever.search(texts, CANDIDATES)
    out = {"dense": evaluate_run({q: _ids(h) for q, h in zip(qids, hits, strict=True)}, ds.qrels)}
    for name, rr in rerankers.items():
        scores = [rr.score(t, h[: max(ks)]) for t, h in zip(texts, hits, strict=True)]
        for k in ks:
            run = {q: _ids(rr.rerank(t, h, k, scores=s))
                   for q, t, h, s in zip(qids, texts, hits, scores, strict=True)}
            out[f"{name}@{k}"] = evaluate_run(run, ds.qrels)
    return out


def _latency(retriever: DenseRetriever, rr: Reranker | None, queries: list[str], k: int) -> dict:
    times = []
    for q in queries:
        t0 = time.perf_counter()
        hits = retriever.search([q], max(k, 10))[0]
        if rr is not None:
            rr.rerank(q, hits, k)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def evaluate(encoder: str = DEFAULT_ENCODER, rerankers: tuple[str, ...] = DEFAULT_RERANKERS,
             ks: tuple[int, ...] = RERANK_KS, root=None, latency_queries: int = 100,
             budget_ms: float = LATENCY_BUDGET_MS) -> tuple[dict, DenseRetriever]:
    dev, test = data.load("train", root), data.load("test", root)
    docs = list(dev.corpus.values())
    retriever = DenseRetriever(encoder, docs)
    models = {name: Reranker(name) for name in rerankers}
    dev_res = _score_split(retriever, dev, models, ks)
    test_res = _score_split(retriever, test, models, ks)

    configs = [{"label": SearchConfig(encoder).label, "reranker": None, "rerank_k": 0,
                "dev": dev_res["dense"], "test": test_res["dense"]}]
    for name in rerankers:
        for k in ks:
            configs.append({"label": SearchConfig(encoder, name, k).label, "reranker": name, "rerank_k": k,
                            "dev": dev_res[f"{name}@{k}"], "test": test_res[f"{name}@{k}"]})

    lat_q = list(dev.queries.values())[:latency_queries]
    _latency(retriever, models[rerankers[0]] if rerankers else None, lat_q[:5], 10)  # warm-up (CUDA init)
    configs[0]["latency"] = _latency(retriever, None, lat_q, 10)
    for c in configs[1:]:
        c["latency"] = _latency(retriever, models[c["reranker"]], lat_q, c["rerank_k"])
    chosen = [configs[0]]
    for name in rerankers:
        mine = [c for c in configs if c["reranker"] == name]
        k, ok = choose_k({c["rerank_k"]: c["dev"]["ndcg@10"] for c in mine}, TIE_TOLERANCE,
                         {c["rerank_k"]: c["latency"]["p95_ms"] for c in mine}, budget_ms)
        chosen.append({**next(c for c in mine if c["rerank_k"] == k), "within_budget": ok})

    from biomedrag.models import device

    report = {
        "dataset": "BEIR SciFact", "corpus_docs": len(docs), "encoder": encoder, "candidates": CANDIDATES,
        "queries": {"dev(train)": len(dev.queries), "test": len(test.queries)},
        "device": device(), "latency_budget_p95_ms": budget_ms,
        "selection": f"rerank_k = best dev nDCG@10 with dev p95 <= {budget_ms:.0f} ms "
                     f"(ties within {TIE_TOLERANCE} -> smaller k)",
        "configs": configs, "chosen": chosen,
    }
    return report, retriever
