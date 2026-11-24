from __future__ import annotations

import csv
import re
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biomedrag import models

DIM = 64


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeEncoder:
    """Hashed bag of words. Unnormalized on purpose, so retrieval must normalize."""

    def encode(self, texts, batch_size=64):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _tokens(t):
                out[i, zlib.crc32(w.encode()) % DIM] += 1.0
        return out


class FakeScorer:
    """Cross-encoder stand-in: the fraction of query words found in the passage."""

    def predict(self, pairs, batch_size=64):
        return np.array([len(set(_tokens(q)) & set(_tokens(p))) / max(1, len(set(_tokens(q))))
                         for q, p in pairs])


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    monkeypatch.setattr(models, "load_encoder", lambda name: FakeEncoder())
    monkeypatch.setattr(models, "load_cross_encoder", lambda name: FakeScorer())
    monkeypatch.setattr(models, "device", lambda: "cpu")


CORPUS = [
    ("d1", "Aspirin and cardiovascular events", "Low dose aspirin reduces myocardial infarction risk."),
    ("d2", "Statins lower cholesterol", "Statin therapy lowers LDL cholesterol in adults."),
    ("d3", "Vitamin D and bone density", "Vitamin D supplementation improves bone mineral density."),
    ("d4", "Smoking and lung cancer", "Tobacco smoking strongly increases lung cancer incidence."),
    ("d5", "Exercise and depression", "Aerobic exercise reduces symptoms of depression."),
    ("d6", "Aspirin side effects", "Aspirin increases gastrointestinal bleeding."),
]
QUERIES = {
    "1": "aspirin reduces myocardial infarction",
    "2": "statin lowers LDL cholesterol",
    "3": "smoking causes lung cancer",
    "4": "exercise improves depression symptoms",
}
QRELS = {"train": {"1": ["d1"], "2": ["d2"]}, "test": {"3": ["d4"], "4": ["d5"]}}


@pytest.fixture
def beir_root(tmp_path, monkeypatch):
    """A tiny dataset in the same layout as the downloaded BEIR SciFact."""
    base = tmp_path / "hf"
    (base / "scifact" / "corpus").mkdir(parents=True)
    (base / "scifact" / "queries").mkdir(parents=True)
    (base / "scifact-qrels").mkdir(parents=True)
    pq.write_table(pa.table({"_id": [c[0] for c in CORPUS], "title": [c[1] for c in CORPUS],
                             "text": [c[2] for c in CORPUS]}), base / "scifact/corpus/corpus.parquet")
    pq.write_table(pa.table({"_id": list(QUERIES), "title": [""] * len(QUERIES), "text": list(QUERIES.values())}),
                   base / "scifact/queries/queries.parquet")
    for split, rels in QRELS.items():
        with open(base / "scifact-qrels" / f"{split}.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["query-id", "corpus-id", "score"])
            for q, docs in rels.items():
                for d in docs:
                    w.writerow([q, d, 1])
    monkeypatch.setenv("BIOMEDRAG_DATA", str(tmp_path))
    return tmp_path
