"""biomedrag download | evaluate [--register] | models | promote V | rollback | search Q | answer Q"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _table(rows: list[dict], cols: list[str]) -> str:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ("-" if v is None else str(v))

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def cmd_evaluate(a) -> None:
    from biomedrag.evaluate import evaluate

    report, retriever = evaluate(a.encoder, tuple(a.rerankers))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    rows = [{"config": c["label"], "dev nDCG@10": c["dev"]["ndcg@10"], **{f"test {k}": v for k, v in c["test"].items()},
             "dev p95 ms": c["latency"]["p95_ms"]} for c in report["configs"]]
    print(_table(rows, ["config", "dev nDCG@10", "test ndcg@10", "test mrr@10", "test recall@10", "test recall@100",
                        "dev p95 ms"]))
    print("\nChosen on dev (latency on " + report["device"] + "):")
    print(_table([{"config": c["label"], "test nDCG@10": c["test"]["ndcg@10"],
                   "p50 ms": c["latency"]["p50_ms"], "p95 ms": c["latency"]["p95_ms"]} for c in report["chosen"]],
                 ["config", "test nDCG@10", "p50 ms", "p95 ms"]))
    if a.register:
        from biomedrag.registry import register
        from biomedrag.retrieve import SearchConfig

        for c in report["chosen"]:
            cfg = SearchConfig(report["encoder"], c["reranker"], c["rerank_k"])
            v = register(cfg, retriever, c["dev"], c["test"], c["latency"],
                         {"dataset": report["dataset"], "device": report["device"]})
            print(f"registered {cfg.label} as version {v}")


def cmd_models(a) -> None:
    from biomedrag.registry import list_versions

    print(_table(list_versions(), ["version", "label", "aliases", "dev_ndcg@10", "test_ndcg@10", "p95_ms",
                                   "promotion"]))


def cmd_promote(a) -> None:
    from biomedrag.registry import promote

    print(json.dumps(promote(a.version, a.min_gain, a.latency_budget_ms, a.override_reason), indent=1))


def cmd_rollback(a) -> None:
    from biomedrag.registry import rollback

    print(json.dumps(rollback(), indent=1))


def cmd_search(a) -> None:
    from biomedrag.registry import load_champion

    for h in load_champion().predict([a.query])[0][: a.k]:
        print(f"{h['score']:>8.3f}  [{h['stage']}] doc {h['doc_id']}: {h['title'][:100]}")


def cmd_answer(a) -> None:
    from biomedrag.generate import answer
    from biomedrag.registry import load_champion

    hits = load_champion().predict([a.query])[0][: a.k]
    print(answer(a.query, hits))


def main(argv: list[str] | None = None) -> None:
    from biomedrag.evaluate import DEFAULT_ENCODER, DEFAULT_RERANKERS

    p = argparse.ArgumentParser(prog="biomedrag", description=__doc__)
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("download", help="fetch BEIR SciFact from the Hugging Face Hub")
    s.set_defaults(func=lambda a: print(__import__("biomedrag.data", fromlist=["download"]).download()))

    s = sub.add_parser("evaluate", help="dense vs dense+cross-encoder on SciFact (dev-chosen, test-reported)")
    s.add_argument("--encoder", default=DEFAULT_ENCODER)
    s.add_argument("--rerankers", nargs="*", default=list(DEFAULT_RERANKERS))
    s.add_argument("--out", type=Path, default=Path("results/scifact_report.json"))
    s.add_argument("--register", action="store_true", help="register each chosen config as a model version")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("models", help="registered versions, aliases and metrics")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("promote", help="move the champion alias if the version passes the gate")
    s.add_argument("version")
    s.add_argument("--min-gain", type=float, default=0.005, help="required dev nDCG@10 gain over champion")
    s.add_argument("--latency-budget-ms", type=float, default=250.0, help="p95 per-query budget")
    s.add_argument("--override-reason", default=None,
                   help="waive only the minimum-gain rule (recorded); never latency or a real regression")
    s.set_defaults(func=cmd_promote)

    s = sub.add_parser("rollback", help="restore the previous champion")
    s.set_defaults(func=cmd_rollback)

    for name, fn, helptext in [("search", cmd_search, "search with the champion model"),
                               ("answer", cmd_answer, "retrieve with the champion, answer with Mistral")]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("query")
        s.add_argument("-k", type=int, default=5)
        s.set_defaults(func=fn)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
