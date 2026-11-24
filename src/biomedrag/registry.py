"""MLflow model registry: every evaluated configuration becomes a model version; the
`champion` alias moves only through a promotion gate.

A registered version is a self-contained pyfunc model. It holds:
* the search config (encoder, re-ranker, rerank_k, top_k),
* the corpus and its precomputed, normalized embeddings,
so `mlflow.pyfunc.load_model("models:/biomed-retriever@champion")` can search without
re-embedding anything. The neural weights are referenced by Hugging Face id, not copied.

Promotion gate (`promote`):
1. **Quality:** dev nDCG@10 must beat the current champion's by at least `min_gain`. The
   gate uses the *dev* split, the same split used for every other choice. Test numbers are
   recorded as tags on the version but never decide anything.
2. **Latency:** p95 per-query latency must be within `latency_budget_ms`.
If there is no champion yet, only the latency check applies. An audited override
(`override_reason`) can waive rule 1's *minimum gain*, for example to replace a champion
built with a known bug. It cannot waive latency or a real dev regression, and it is
recorded as `promoted_override`. On promotion, the old champion
gets the alias `previous`, so `rollback` is one call. Every decision is written to the
version's tags (`promotion`, `promotion_reason`), so the registry records its own history.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient

from biomedrag.data import Doc, data_root
from biomedrag.retrieve import DenseRetriever, SearchConfig, Searcher

MODEL_NAME = "biomed-retriever"
EXPERIMENT = "biomed-retrieval"
CHAMPION, PREVIOUS = "champion", "previous"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{data_root() / 'mlflow.db'}")


def _setup() -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_registry_uri(tracking_uri())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        root = Path(os.environ.get("BIOMEDRAG_ARTIFACTS", data_root() / "mlartifacts"))
        root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT, artifact_location=root.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient()


class RetrieverModel(mlflow.pyfunc.PythonModel):
    """pyfunc: list of query strings (or a DataFrame with a `query` column) -> top-k hits."""

    def load_context(self, context) -> None:
        cfg = json.loads(Path(context.artifacts["config"]).read_text())
        docs = [Doc(**d) for d in json.loads(Path(context.artifacts["docs"]).read_text())]
        emb = np.load(context.artifacts["embeddings"])
        self.searcher = Searcher(SearchConfig(**cfg), DenseRetriever(cfg["encoder"], docs, emb))

    def predict(self, context, model_input, params=None):
        queries = list(model_input["query"]) if hasattr(model_input, "columns") else list(model_input)
        return [[{"doc_id": h.doc.id, "title": h.doc.title, "text": h.doc.text,
                  "score": round(h.score, 4), "stage": h.stage}
                 for h in self.searcher.search(q)] for q in queries]


def _flat(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{k.replace('@', '_at_')}": v for k, v in metrics.items()}


def register(cfg: SearchConfig, retriever: DenseRetriever, dev: dict, test: dict, latency: dict,
             extra_tags: dict | None = None) -> str:
    """Log one evaluated configuration as an MLflow run and register it as a new version."""
    client = _setup()
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name=cfg.label) as run:
        tmp = Path(tmp)
        (tmp / "config.json").write_text(json.dumps(asdict(cfg)))
        (tmp / "docs.json").write_text(json.dumps([asdict(d) for d in retriever.docs]))
        np.save(tmp / "embeddings.npy", retriever.embeddings)
        mlflow.log_params({k: str(v) for k, v in asdict(cfg).items()})
        mlflow.log_metrics({**_flat("dev", dev), **_flat("test", test),
                            "latency_p50_ms": latency["p50_ms"], "latency_p95_ms": latency["p95_ms"]})
        mlflow.set_tags({"label": cfg.label, **(extra_tags or {})})
        info = mlflow.pyfunc.log_model(
            name="model", python_model=RetrieverModel(),
            artifacts={"config": str(tmp / "config.json"), "docs": str(tmp / "docs.json"),
                       "embeddings": str(tmp / "embeddings.npy")},
            pip_requirements=["sentence-transformers>=3.0", "faiss-cpu>=1.8", "numpy>=1.26", "torch"],
            registered_model_name=MODEL_NAME,
        )
        version = str(info.registered_model_version)
    client.set_model_version_tag(MODEL_NAME, version, "label", cfg.label)
    client.set_model_version_tag(MODEL_NAME, version, "test_ndcg_at_10", f"{test['ndcg@10']:.4f}")
    client.update_model_version(MODEL_NAME, version, description=(
        f"{cfg.label}: dev nDCG@10 {dev['ndcg@10']:.4f}, test nDCG@10 {test['ndcg@10']:.4f}, "
        f"p95 {latency['p95_ms']:.0f} ms (run {run.info.run_id})"))
    return version


def _metrics(client: MlflowClient, version: str) -> dict:
    mv = client.get_model_version(MODEL_NAME, version)
    return client.get_run(mv.run_id).data.metrics


def champion_version(client: MlflowClient | None = None) -> str | None:
    client = client or _setup()
    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version)
    except mlflow.exceptions.MlflowException:
        return None


def promote(version: str, min_gain: float = 0.005, latency_budget_ms: float = 250.0,
            override_reason: str | None = None) -> dict:
    """`override_reason` waives *only* the minimum-gain rule (e.g. replacing a champion built
    with a known bug by the same config with the bug fixed). It never waives the latency budget
    or a dev regression larger than `min_gain`, and the reason is recorded on the version."""
    version = str(version)  # MLflow returns versions as int from some calls and str from others
    client = _setup()
    cand = _metrics(client, version)
    current = champion_version(client)
    reasons, waived = [], None
    if cand["latency_p95_ms"] > latency_budget_ms:  # never waivable
        reasons.append(f"p95 latency {cand['latency_p95_ms']:.0f} ms > budget {latency_budget_ms:.0f} ms")
    if current is not None and current != version:
        champ = _metrics(client, current)
        gain = cand["dev_ndcg_at_10"] - champ["dev_ndcg_at_10"]
        if gain < min_gain:
            msg = f"dev nDCG@10 gain {gain:+.4f} vs champion v{current} < {min_gain}"
            if override_reason and gain > -min_gain:  # a waiver may not hide a real regression
                waived = msg
            else:
                reasons.append(msg)
    elif current == version:
        reasons.append("already champion")
    promoted = not reasons
    if promoted:
        if current is not None:
            client.set_registered_model_alias(MODEL_NAME, PREVIOUS, current)
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
        if waived:
            reason = (f"OVERRIDE ({override_reason}); waived: {waived}; "
                      f"p95 {cand['latency_p95_ms']:.0f} ms within budget")
        else:
            reason = (f"dev nDCG@10 {cand['dev_ndcg_at_10']:.4f}"
                      + (f" beats v{current}" if current else " (first champion)")
                      + f", p95 {cand['latency_p95_ms']:.0f} ms within budget")
    else:
        reason = "; ".join(reasons)
    status = ("promoted_override" if waived else "promoted") if promoted else "rejected"
    client.set_model_version_tag(MODEL_NAME, version, "promotion", status)
    client.set_model_version_tag(MODEL_NAME, version, "promotion_reason", reason)
    return {"version": version, "promoted": promoted, "reason": reason, "previous_champion": current}


def rollback() -> dict:
    client = _setup()
    try:
        prev = str(client.get_model_version_by_alias(MODEL_NAME, PREVIOUS).version)
    except mlflow.exceptions.MlflowException as exc:
        raise ValueError("no `previous` alias to roll back to") from exc
    current = champion_version(client)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, prev)
    if current is not None:
        client.set_model_version_tag(MODEL_NAME, current, "promotion", "rolled_back")
    client.delete_registered_model_alias(MODEL_NAME, PREVIOUS)
    return {"champion": prev, "rolled_back": current}


def list_versions() -> list[dict]:
    client = _setup()
    aliases: dict[str, list[str]] = {}
    for alias in (CHAMPION, PREVIOUS):
        try:
            aliases.setdefault(str(client.get_model_version_by_alias(MODEL_NAME, alias).version), []).append(alias)
        except mlflow.exceptions.MlflowException:
            pass
    out = []
    for mv in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda m: int(m.version)):
        m = client.get_run(mv.run_id).data.metrics
        out.append({"version": str(mv.version), "label": mv.tags.get("label"),
                    "aliases": aliases.get(str(mv.version), []),
                    "dev_ndcg@10": m.get("dev_ndcg_at_10"), "test_ndcg@10": m.get("test_ndcg_at_10"),
                    "p95_ms": m.get("latency_p95_ms"), "promotion": mv.tags.get("promotion", "-")})
    return out


def load_champion():
    _setup()
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")
