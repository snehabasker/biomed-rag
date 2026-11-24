# biomed-rag: working agreement for AI coding assistants

I set the design and the acceptance checks. The assistant (Claude Code) writes code within
these rules, and nothing counts as done until the checks pass.

## Commands (WSL2 / Linux; venv at ~/venvs/biomedrag)
- Install: `uv pip install -e ".[dev,llm]"` (plus a CUDA 12.8 PyTorch wheel for RTX 50xx GPUs)
- Data: `biomedrag download` (BEIR SciFact from the Hugging Face Hub, into $BIOMEDRAG_DATA)
- Test: `python -m pytest`. Lint: `ruff check src tests`.
- Evaluate: `biomedrag evaluate [--register]`. Registry: `biomedrag models | promote <v> | rollback`.

## Definition of done
1. Tests are added or updated for the new behaviour, including an edge case. Tests never hit
   the network: models load only through `biomedrag/models.py`, and tests replace it.
2. `pytest` and `ruff check src tests` both pass.
3. Every bug that gets fixed gets a regression test (for example L2 vs cosine).
4. A change to retrieval, re-ranking or metrics is re-run on SciFact, and the README tables are
   updated from `results/scifact_report.json`.

## Evaluation integrity
- **Choices are made on dev (SciFact train queries), results are reported on test.** That
  covers `rerank_k`, model choice and the promotion gate. Never tune on test.
- Metrics follow trec_eval/BEIR conventions. If the dense baseline moves far from the
  published all-MiniLM-L6-v2 figure on SciFact (nDCG@10 ≈ 0.645), check the code before
  trusting any other number.
- Latency numbers always say which device they were measured on.

## Model registry
- Every registered version carries its dev/test metrics and latency in its MLflow run.
- `champion` moves **only** through `biomedrag promote` (dev quality gate plus latency budget).
  Never set aliases by hand. The previous champion keeps the alias `previous` for rollback.
- Every promotion decision, promoted or rejected, is written to the version's tags.

## Reporting
- The README states only what was run. Generation (Mistral) is optional and not benchmarked;
  say so.
- Report negative or neutral results too (for example, a re-ranker that does not help).
