"""BEIR SciFact: 5,183 PubMed-derived abstracts, scientific claims as queries, expert labels.

Splits: `train` (809 queries) is used as the development split for choices such as how many
candidates to re-rank, and `test` (300 queries) is only reported. The data is downloaded
from the Hugging Face Hub (BeIR/scifact, BeIR/scifact-qrels), not redistributed here.
SciFact is CC BY-NC 2.0 (Wadden et al., 2020).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


def data_root() -> Path:
    return Path(os.environ.get("BIOMEDRAG_DATA", Path.home() / "data" / "biomedrag"))


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str

    @property
    def passage(self) -> str:
        """What both the bi-encoder and the cross-encoder see."""
        return f"{self.title}. {self.text}" if self.title else self.text


@dataclass
class Dataset:
    corpus: dict[str, Doc]
    queries: dict[str, str]  # only queries that have labels in this split
    qrels: dict[str, dict[str, int]]  # query id -> {doc id: relevance}


def download(root: Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    root = (root or data_root()) / "hf"
    for repo in ("BeIR/scifact", "BeIR/scifact-qrels"):
        snapshot_download(repo, repo_type="dataset", local_dir=root / repo.split("/")[1], token=False)
    return root


def load(split: str, root: Path | None = None) -> Dataset:
    if split not in ("train", "test"):
        raise ValueError(f"split must be train or test, not {split!r}")
    base = (root or data_root()) / "hf"
    corpus_rows = pq.read_table(next((base / "scifact" / "corpus").glob("*.parquet"))).to_pylist()
    corpus = {str(r["_id"]): Doc(str(r["_id"]), r["title"] or "", r["text"] or "") for r in corpus_rows}
    qrels: dict[str, dict[str, int]] = {}
    with open(base / "scifact-qrels" / f"{split}.tsv", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    query_rows = pq.read_table(next((base / "scifact" / "queries").glob("*.parquet"))).to_pylist()
    queries = {str(r["_id"]): r["text"] for r in query_rows if str(r["_id"]) in qrels}
    missing = set(qrels) - set(queries)
    if missing:
        raise ValueError(f"{len(missing)} labelled queries have no text, e.g. {sorted(missing)[:3]}")
    return Dataset(corpus, queries, qrels)
