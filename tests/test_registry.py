"""MLflow registry and the promotion gate, against a throwaway local SQLite registry."""

import pytest

from biomedrag import data, registry
from biomedrag.retrieve import DenseRetriever, SearchConfig


@pytest.fixture
def reg(beir_root, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("BIOMEDRAG_ARTIFACTS", str(tmp_path / "artifacts"))
    ds = data.load("test", beir_root)
    return DenseRetriever("fake", list(ds.corpus.values()))


def _register(retriever, label_k: int, dev_ndcg: float, p95: float) -> str:
    cfg = SearchConfig("fake", "fake-ce" if label_k else None, label_k)
    m = {"ndcg@10": dev_ndcg, "mrr@10": dev_ndcg, "recall@10": 0.9, "recall@100": 0.95}
    return registry.register(cfg, retriever, m, {**m, "ndcg@10": dev_ndcg - 0.01},
                             {"p50_ms": p95 / 2, "p95_ms": p95})


def test_gate_promotes_rejects_and_rolls_back(reg):
    v1 = _register(reg, 0, dev_ndcg=0.60, p95=10)
    assert registry.promote(v1)["promoted"]  # first champion: only latency is checked
    assert registry.champion_version() == v1

    v2 = _register(reg, 20, dev_ndcg=0.602, p95=40)  # +0.002 < min_gain
    d2 = registry.promote(v2, min_gain=0.005)
    assert not d2["promoted"] and "gain" in d2["reason"] and registry.champion_version() == v1

    v3 = _register(reg, 100, dev_ndcg=0.70, p95=900)  # much better but too slow
    d3 = registry.promote(v3, latency_budget_ms=250)
    assert not d3["promoted"] and "latency" in d3["reason"]

    v4 = _register(reg, 50, dev_ndcg=0.68, p95=120)
    d4 = registry.promote(v4)
    assert d4["promoted"] and d4["previous_champion"] == v1 and registry.champion_version() == v4

    rows = {r["version"]: r for r in registry.list_versions()}
    assert rows[v4]["aliases"] == ["champion"] and rows[v1]["aliases"] == ["previous"]
    assert rows[v2]["promotion"] == "rejected" and rows[v3]["promotion"] == "rejected"

    assert registry.rollback() == {"champion": v1, "rolled_back": v4}
    assert registry.champion_version() == v1
    with pytest.raises(ValueError):
        registry.rollback()  # `previous` was consumed


def test_champion_model_loads_and_searches_without_reembedding(reg, monkeypatch):
    v = _register(reg, 4, dev_ndcg=0.5, p95=10)
    registry.promote(v)
    model = registry.load_champion()
    out = model.predict(["smoking lung cancer", "statin LDL cholesterol"])
    assert out[0][0]["doc_id"] == "d4" and out[1][0]["doc_id"] == "d2"
    assert out[0][0]["stage"] == "rerank" and "text" in out[0][0]
