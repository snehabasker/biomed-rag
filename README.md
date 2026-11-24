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

These are the final numbers: fp16 cross-encoders on an RTX 5050 laptop GPU, re-ranking the
top-k of 100 dense candidates. Latency is p95 per query (dense search + re-ranking),
measured on dev queries.

| configuration | nDCG@10 | MRR@10 | Recall@10 | p95 latency |
|---|---:|---:|---:|---:|
| dense (all-MiniLM-L6-v2) | 0.648 | 0.607 | 0.788 | 11 ms |
| + ms-marco-MiniLM-L6 re-ranker, k = 20 | 0.689 | 0.658 | 0.812 | 58 ms |
| + ms-marco-MiniLM-L6 re-ranker, k = 100 | 0.687 | 0.656 | 0.809 | 234 ms |
| + MedCPT re-ranker, k = 10 | 0.719 | 0.702 | 0.788 | 87 ms |
| **+ MedCPT re-ranker, k = 20 (current champion)** | **0.742** | **0.721** | **0.828** | **215 ms** |
| + MedCPT re-ranker, k = 50 | 0.758 | 0.734 | 0.853 | 475 ms |
| + MedCPT re-ranker, k = 100 | 0.768 | 0.742 | 0.867 | 872 ms |

Recall@100 is 0.925 for every row, because re-ranking only reorders the same 100 candidates.

- **The metric code is validated.** The dense baseline (0.648) matches the published
  all-MiniLM-L6-v2 score on SciFact (≈ 0.645).
- **A domain-specific re-ranker matters.** MedCPT, which NCBI trained on PubMed search logs,
  adds up to +0.12 nDCG@10. The general MS MARCO re-ranker adds only +0.04, and gains
  nothing past k = 20.
- **Quality vs latency is the real decision.** Within a 250 ms p95 budget, the best choice
  is MedCPT with k = 20: +0.094 nDCG@10 (+14% relative) over dense. k = 100 would add a
  further +0.026, at 4x the latency.
- **fp16 vs fp32:** identical ranking quality (to 3 decimals) and 3.3x faster for MedCPT@100
  (2,187 ms → 668 ms in the same run setup).

### How the champion was chosen: the registry history

| version | configuration | what happened | gate decision |
|---|---|---|---|
| v1 | dense, fp32 | first registration | **promoted** (first champion) |
| v2, v3 | ms-marco@100, MedCPT@100, fp32, k chosen by quality only | 466 ms and 2,187 ms p95 | rejected: over the 250 ms budget |
| v4–v6 | fp16, k chosen within the latency budget | MedCPT@20: dev +0.088 over v1, 153 ms | **v6 promoted**; v4/v5 rejected (not better) |
| v7–v9 | same, after fixing the sigmoid-tie bug (below) | MedCPT@20: dev +0.0005 over v6 | v9 rejected (gain < 0.005), then **promoted by audited override** |

The override is deliberate and narrow. It waives only the minimum-gain rule, never the
latency budget or a real regression, and the reason is stored on the version:
`OVERRIDE (v6 was scored with the fp16 sigmoid-tie bug; v9 is the same config with raw-logit
scoring)`. `v6` keeps the `previous` alias, so rolling back is one command.

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
biomedrag promote 9                      # passes the gate -> champion, else rejected with a reason
biomedrag promote 9 --override-reason "..."   # waive only min-gain; recorded on the version
biomedrag rollback                       # champion -> previous
biomedrag search "statins reduce LDL cholesterol"
MISTRAL_API_KEY=... biomedrag answer "does aspirin reduce myocardial infarction risk?"
pytest                                   # 14 tests, no network (fake models), temp MLflow registry
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
  gate caught,
- **a ranking bug from sigmoid saturation in fp16.** The cross-encoder's default sigmoid
  turned every logit above ~8 into exactly 1.0. On a 100-query dev sample, 52 queries had
  tied scores (6 at rank 1), and the ties silently fell back to dense order. It was spotted
  because the champion's top 3 all scored "1.000". The fix ranks on raw logits, with a
  regression test; MedCPT@20 test nDCG@10 went from 0.739 to 0.742.

## Data and licences

Code: MIT. BEIR SciFact (Wadden et al., 2020) is CC BY-NC 2.0. It is downloaded at runtime
and not redistributed. MedCPT: Jin et al., 2023 (NCBI).
