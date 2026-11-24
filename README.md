# biomed-rag

**Retrieval for biomedical RAG: dense retrieval + cross-encoder re-ranking, measured on a
public benchmark, with an MLflow model registry that only promotes a model through a
quality and latency gate.**

A RAG answer can only cite what retrieval found, so this repo measures and improves the
retrieval stage:

1. **Dense retrieval:** `all-MiniLM-L6-v2` embeddings, cosine similarity, FAISS.
2. **Cross-encoder re-ranking** of the top-k candidates, with a general model
   (`cross-encoder/ms-marco-MiniLM-L6-v2`) and a biomedical one (`ncbi/MedCPT-Cross-Encoder`).
3. **Evaluation on BEIR SciFact:** 5,183 biomedical abstracts with expert-labelled
   scientific claims as queries. Settings are chosen on the 809 train (dev) queries and
   reported on the 300 test queries.
4. **MLflow registry:** every evaluated configuration becomes a model version. The
   `champion` alias moves only when a version beats the current champion on dev nDCG@10
   *and* fits the p95 latency budget. Rollback is one command.
5. **Answering (optional):** Mistral answers from the champion's results and cites
   `[doc:<id>]`. This step isn't benchmarked here.

## Results (BEIR SciFact test set, 300 queries)

First run: fp32 on an RTX 5050 laptop GPU, re-ranking the top 100 dense candidates.

| configuration | nDCG@10 | MRR@10 | Recall@10 | p95 latency / query |
|---|---:|---:|---:|---:|
| dense (all-MiniLM-L6-v2) | 0.648 | 0.607 | 0.788 | 15 ms |
| + ms-marco-MiniLM-L6 re-ranker, k = 20 | 0.689 | 0.658 | 0.812 | – |
| + ms-marco-MiniLM-L6 re-ranker, k = 100 | 0.687 | 0.656 | 0.809 | 466 ms |
| + MedCPT re-ranker, k = 20 | 0.739 | 0.717 | 0.828 | – |
| + MedCPT re-ranker, k = 50 | 0.756 | 0.730 | 0.853 | – |
| **+ MedCPT re-ranker, k = 100** | **0.765** | **0.738** | **0.867** | 2,187 ms |

Recall@100 is 0.925 for every row, because re-ranking only reorders the same 100 candidates.

- **The metric code is validated.** The dense baseline (0.648) matches the published
  all-MiniLM-L6-v2 score on SciFact (≈ 0.645).
- **A domain-specific re-ranker matters.** MedCPT, which NCBI trained on PubMed search logs,
  adds +0.117 nDCG@10 (+18% relative). The general MS MARCO re-ranker adds only +0.04.
- **More candidates help MedCPT but cost latency.** Going from k = 20 to 100 adds +0.026
  nDCG@10 at roughly 5x the cost.

### The promotion gate in action

| version | configuration | dev nDCG@10 | p95 | gate decision |
|---|---|---:|---:|---|
| v1 | dense | 0.660 | 15 ms | **promoted** (first champion) |
| v2 | + ms-marco @100 | 0.715 | 466 ms | rejected: p95 466 ms > budget 250 ms |
| v3 | + MedCPT @100 | 0.779 | 2,187 ms | rejected: p95 2,187 ms > budget 250 ms |

The most accurate model was about 9x over the latency budget, and the gate kept it out of
production. The fix is the engineering work a team would actually do:
- **fp16 inference** for the cross-encoder,
- **choosing k by the best dev quality that fits the budget,** instead of by quality alone.

**Status:** the fp16 run with latency-aware k selection is in progress. Its results, and
which version the gate promotes, will be added here.

## How it works

```
query ─► bi-encoder ─► FAISS (cosine) ─► top 100 ─► cross-encoder on top k ─► top 10 ─► (Mistral answer)
              stage 1: fast, recall              stage 2: accurate, costly
```

- **Why two stages.** A bi-encoder embeds query and abstract separately, so abstracts are
  embedded once and search is one matrix product. A cross-encoder reads query and abstract
  *together* with full attention, which is far more accurate but needs one transformer pass
  per pair. So it only re-orders the top k.
- **Fixed from the original app.** The original app searched with L2 distance on
  unnormalized vectors, so a relevant long abstract with a large embedding norm could lose
  to an off-topic short one. It also dropped results below an arbitrary `1/(1+d) >= 0.75`
  cutoff. This repo uses normalized vectors with inner product (cosine) and no cutoff. A
  regression test covers the L2 case.
- **The registry entry is self-contained.** Each version stores its config, the corpus and
  the precomputed normalized embeddings, so
  `mlflow.pyfunc.load_model("models:/biomed-retriever@champion")` searches straight away.
  Model weights are referenced by Hugging Face id.

## Usage

```bash
uv pip install -e ".[dev,llm]"           # plus a CUDA 12.8 PyTorch wheel for RTX 50xx GPUs
biomedrag download                       # BEIR SciFact from the Hugging Face Hub
biomedrag evaluate --register            # all configs; register the dev-chosen ones
biomedrag models                         # versions, aliases, metrics, gate decisions
biomedrag promote 3                      # passes the gate -> champion, else rejected with a reason
biomedrag rollback                       # champion -> previous
biomedrag search "statins reduce LDL cholesterol"
MISTRAL_API_KEY=... biomedrag answer "does aspirin reduce myocardial infarction risk?"
pytest                                   # 12 tests, no network (fake models), temp MLflow registry
```

## Layout

```
src/biomedrag/
  data.py      SciFact loading (dev = train queries, test = test queries)
  models.py    the only place models are loaded (tests replace it)
  retrieve.py  DenseRetriever (cosine/FAISS), Reranker (cross-encoder on top k), Searcher
  metrics.py   nDCG@10, MRR@10, Recall@k (trec_eval conventions)
  evaluate.py  one retrieval pass and one scoring pass per re-ranker, reused for every k
  registry.py  MLflow pyfunc model, register / promote (gate) / rollback / list
  generate.py  Mistral answer with [doc:id] citations (optional)
results/       evaluation reports (JSON) and logs
```

## How this was built

I built this with Claude Code as a pair programmer. I chose the benchmark, the dev/test
discipline, the promotion rules and the acceptance checks, all written down in
[CLAUDE.md](CLAUDE.md). The assistant wrote most of the code within those rules. Results
were accepted only after checks such as the baseline matching its published score. Issues
caught along the way:
- the L2 distance bug in the original app,
- MLflow returning versions as `int` from some calls and `str` from others, which would have
  silently broken the "is this already champion?" comparison,
- a quality-only choice of k that produced a model 9x over the latency budget, which the
  gate caught.

## Data and licences

Code: MIT. BEIR SciFact (Wadden et al., 2020) is CC BY-NC 2.0. It is downloaded at runtime
and not redistributed. MedCPT: Jin et al., 2023 (NCBI).
