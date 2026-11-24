import numpy as np
import pytest

from biomedrag import data
from biomedrag.data import Doc
from biomedrag.evaluate import choose_k, evaluate
from biomedrag.generate import answer, build_prompt
from biomedrag.metrics import evaluate_run, mrr_at_k, ndcg_at_k, recall_at_k
from biomedrag.retrieve import DenseRetriever, Hit, Reranker, SearchConfig, Searcher


def test_metric_values_match_hand_computation():
    rel = {"a": 1}
    assert ndcg_at_k(["a", "b"], rel, 10) == 1.0
    assert ndcg_at_k(["b", "a"], rel, 10) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k(["b", "c"], rel, 10) == 0.0
    assert recall_at_k(["x", "a"], {"a": 1, "b": 1}, 1) == 0.0
    assert recall_at_k(["x", "a"], {"a": 1, "b": 1}, 2) == 0.5
    assert mrr_at_k(["x", "y", "a"], rel, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "a"], rel, 1) == 0.0
    assert evaluate_run({}, {"q": rel})["ndcg@10"] == 0.0  # unanswered query counts as 0


def test_cosine_not_l2_large_norm_relevant_doc_still_wins():
    """Regression for the original app: L2 on unnormalized vectors ranked a weakly related
    small-norm document above the relevant long abstract."""
    q, relevant, offtopic = np.array([1.0, 0.0]), np.array([5.0, 0.2]), np.array([0.6, 0.6])
    assert np.linalg.norm(q - offtopic) < np.linalg.norm(q - relevant)  # L2 gets it wrong

    class Fixed:
        def encode(self, texts, batch_size=64):
            table = {"q": q, "relevant": relevant, "offtopic": offtopic}
            return np.stack([table[t.split(". ")[-1]] for t in texts]).astype(np.float32)

    docs = [Doc("r", "", "relevant"), Doc("o", "", "offtopic")]
    r = DenseRetriever("fixed", docs, Fixed().encode([d.passage for d in docs]))
    r.encoder = Fixed()  # queries go through the same fixed vectors
    assert [h.doc.id for h in r.search(["q"], 2)[0]] == ["r", "o"]


def _hits(ids):
    return [Hit(Doc(i, "", text), 1.0 - n * 0.1, "dense") for n, (i, text) in enumerate(ids)]


def test_reranker_reorders_only_the_head():
    hits = _hits([("a", "unrelated"), ("b", "statin cholesterol"), ("c", "nothing"), ("d", "statin LDL cholesterol")])
    out = Reranker("fake").rerank("statin LDL cholesterol", hits, rerank_k=3)
    assert [h.doc.id for h in out] == ["b", "a", "c", "d"]  # "d" is outside k=3 and keeps its place
    assert {h.stage for h in out[:3]} == {"rerank"} and out[3].stage == "dense"
    full = Reranker("fake").rerank("statin LDL cholesterol", hits, rerank_k=4)
    assert full[0].doc.id == "d"


def test_precomputed_scores_give_same_order_as_fresh_scoring():
    hits = _hits([("a", "x"), ("b", "statin"), ("c", "statin ldl")])
    rr = Reranker("fake")
    s = rr.score("statin ldl", hits)
    assert [h.doc.id for h in rr.rerank("statin ldl", hits, 2, scores=s)] == \
           [h.doc.id for h in rr.rerank("statin ldl", hits, 2)]


def test_choose_k_prefers_smaller_k_within_tolerance():
    assert choose_k({10: 0.700, 20: 0.704, 50: 0.705}) == (20, True)  # 0.704 within 0.002 of best
    assert choose_k({10: 0.690, 20: 0.701, 50: 0.7015}) == (20, True)


def test_choose_k_respects_the_latency_budget():
    ndcg = {10: 0.72, 20: 0.75, 50: 0.76, 100: 0.78}
    p95 = {10: 60.0, 20: 120.0, 50: 300.0, 100: 600.0}
    assert choose_k(ndcg, p95_ms=p95, budget_ms=250) == (20, True)  # best quality that fits
    assert choose_k(ndcg, p95_ms=p95, budget_ms=10) == (10, False)  # nothing fits: smallest k, flagged


def test_load_rejects_unknown_split_and_keeps_only_labelled_queries(beir_root):
    ds = data.load("train", beir_root)
    assert set(ds.queries) == {"1", "2"} and len(ds.corpus) == 6
    with pytest.raises(ValueError):
        data.load("dev", beir_root)


def test_evaluate_end_to_end_on_tiny_dataset(beir_root):
    report, _ = evaluate("fake-enc", ("fake-ce",), ks=(2, 4), latency_queries=2)
    labels = [c["label"] for c in report["configs"]]
    assert labels == ["dense", "dense+fake-ce@2", "dense+fake-ce@4"]
    assert report["queries"] == {"dev(train)": 2, "test": 2}
    chosen = report["chosen"]
    assert [c["reranker"] for c in chosen] == [None, "fake-ce"]
    assert all("p95_ms" in c["latency"] for c in report["configs"])
    assert chosen[1]["within_budget"]
    assert all(0.0 <= c["test"]["ndcg@10"] <= 1.0 for c in report["configs"])


def test_searcher_returns_top_k(beir_root):
    ds = data.load("test", beir_root)
    s = Searcher(SearchConfig("fake", "fake-ce", rerank_k=4, top_k=2),
                 DenseRetriever("fake", list(ds.corpus.values())))
    hits = s.search("smoking lung cancer")
    assert len(hits) == 2 and hits[0].doc.id == "d4"


class FakeMistral:
    def __init__(self):
        self.prompts = []
        self.chat = self

    def complete(self, model, temperature, messages):
        self.prompts.append(messages[-1]["content"])

        message = type("M", (), {"content": "Aspirin helps [doc:d1]."})
        return type("R", (), {"choices": (type("C", (), {"message": message}),)})


def test_answer_cites_retrieved_docs_and_refuses_without_evidence():
    hits = [{"doc_id": "d1", "title": "Aspirin", "text": "reduces MI"}]
    client = FakeMistral()
    assert "[doc:d1]" in answer("does aspirin help?", hits, client=client)
    assert "[doc:d1] Aspirin" in client.prompts[0]
    assert "no evidence" in answer("anything", [], client=client)
    assert len(build_prompt("q", hits * 100, max_chars=200)) < 400
