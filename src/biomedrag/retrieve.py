"""Two-stage retrieval: a dense bi-encoder finds candidates, a cross-encoder re-orders them.

Stage 1 (bi-encoder): query and passages are embedded separately, so passage vectors are
computed once and search is a single matrix product. That is fast, but the query never
"sees" the passage. Vectors are L2-normalized and searched by inner product, which is
cosine similarity. The original app used L2 distance on unnormalized vectors, where long
abstracts with large norms are pushed away regardless of topic, and then dropped results
below a fixed `1/(1+d) >= 0.75` cutoff that has no calibrated meaning.

Stage 2 (cross-encoder): each (query, passage) pair is scored jointly with full attention
between them. That is much more accurate, but it costs one transformer pass per pair, so it
only runs on the top `rerank_k` candidates. Candidates below `rerank_k` keep their stage-1
order, so recall@N for N >= rerank_k is unchanged by re-ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from biomedrag import models
from biomedrag.data import Doc


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class Hit:
    doc: Doc
    score: float
    stage: str  # "dense" or "rerank"


class DenseRetriever:
    def __init__(self, encoder_name: str, docs: list[Doc], embeddings: np.ndarray | None = None):
        self.encoder_name = encoder_name
        self.docs = docs
        self.encoder = models.load_encoder(encoder_name)
        if embeddings is None:
            embeddings = self.encoder.encode([d.passage for d in docs])
        self.embeddings = _normalize(embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, queries: list[str], k: int) -> list[list[Hit]]:
        q = _normalize(self.encoder.encode(queries))
        scores, idx = self.index.search(q, min(k, len(self.docs)))
        return [[Hit(self.docs[i], float(s), "dense") for s, i in zip(row_s, row_i, strict=True) if i >= 0]
                for row_s, row_i in zip(scores, idx, strict=True)]


class Reranker:
    def __init__(self, model_name: str, batch_size: int = 128):
        self.model_name = model_name
        self.batch_size = batch_size
        self.scorer = models.load_cross_encoder(model_name)

    def score(self, query: str, hits: list[Hit]) -> np.ndarray:
        if not hits:
            return np.zeros(0)
        return self.scorer.predict([(query, h.doc.passage) for h in hits], batch_size=self.batch_size)

    def rerank(self, query: str, hits: list[Hit], rerank_k: int,
               scores: np.ndarray | None = None) -> list[Hit]:
        """Re-order the first `rerank_k` hits by cross-encoder score; keep the rest as they are.
        `scores` (for those hits) can be passed in to reuse one scoring pass across several k."""
        head, tail = hits[:rerank_k], hits[rerank_k:]
        s = self.score(query, head) if scores is None else np.asarray(scores[: len(head)])
        order = np.argsort(-s, kind="stable")
        return [Hit(head[i].doc, float(s[i]), "rerank") for i in order] + tail


@dataclass(frozen=True)
class SearchConfig:
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker: str | None = None
    rerank_k: int = 0
    top_k: int = 10

    @property
    def label(self) -> str:
        if not self.reranker:
            return "dense"
        return f"dense+{self.reranker.split('/')[-1]}@{self.rerank_k}"


class Searcher:
    def __init__(self, cfg: SearchConfig, retriever: DenseRetriever):
        self.cfg = cfg
        self.retriever = retriever
        self.reranker = Reranker(cfg.reranker) if cfg.reranker else None

    def search(self, query: str) -> list[Hit]:
        k = max(self.cfg.top_k, self.cfg.rerank_k)
        hits = self.retriever.search([query], k)[0]
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits, self.cfg.rerank_k)
        return hits[: self.cfg.top_k]
