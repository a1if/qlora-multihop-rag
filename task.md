# Task Tracker: Resume Upgrade

Branch: `eval-hotpotqa` · PR #1 open, CI green · Tests: 57 passing · Lint: clean

Goal: show how far a small open LLM (DeepSeek-R1-Distill-Qwen-1.5B) can be improved for multi-hop QA through fine-tuning, measured on HotpotQA, with the engineering around it (training pipeline, eval harness, serving own checkpoints, tests, CI). No cloud/hosted models: every model in the results is one we trained or its base.

Legend: `[x]` done · `[ ]` todo · `[~]` written but not checked against real models/data

---

## Files

### Source: new
| File | Status | Purpose |
|---|---|---|
| `src/llm.py` | [~] | `LLM` protocol; `HFLLM` (in-process) and `OpenAICompatibleLLM` (own checkpoint served by vLLM / Ollama); `make_llm(spec)` |
| `src/retriever.py` | [x] | FAISS inner-product retriever over normalised embeddings, save/load, round-robin evidence merge |
| `src/text_utils.py` | [x] | `<think>` stripping; answer / query / sub-question parsing |
| `src/metrics.py` | [x] | HotpotQA EM/F1, support recall, bootstrap CI, paired-bootstrap delta |
| `src/hotpotqa.py` | [~] | Loads HotpotQA distractor dev set and pools its paragraphs into one corpus (dataset ID and fields not yet checked against the real dataset) |
| `src/eval_hotpotqa.py` | [~] | Ablation benchmark CLI: model specs, concurrent workers, resumable JSONL, summary.md/csv |
| `src/api.py` | [~] | FastAPI service: `/v1/answer`, `/healthz`, `/readyz`, `/metrics`, request IDs, JSON logs |
| `src/build_index.py` | [~] | Index build CLI (arXiv or HotpotQA) |

### Source: modified
| File | Status | Change |
|---|---|---|
| `src/rag_pipeline.py` | [~] | Rewritten: 4 modes (closed / single / iterative / decompose), same passage budget for all, per-thread stage timings; fixed prompt-in-output, L2/cosine mismatch and sampling bugs |
| `src/app.py` | [~] | Gradio demo on the new pipeline, configured from env vars |
| `configs/training_config.py` | [x] | `RAGConfig` gains `llm` spec, prompt style, reasoning, generation settings; GPU-specific comments removed |
| `src/finetune.py` | [ ] | Unchanged; still Alpaca training (see Phase 2) |
| `src/evaluate.py` | [x] | **Deleted**: graded RAG against another model's output, not correct answers |

### Tests
| File | Status | Covers |
|---|---|---|
| `tests/conftest.py` | [x] | Fake embedder, scripted fake LLM, toy corpus |
| `tests/test_text_utils.py` | [x] | Parsing edge cases |
| `tests/test_metrics.py` | [x] | EM/F1, recall, CIs |
| `tests/test_retriever.py` | [x] | Ranking, cosine scores, save/load, merge |
| `tests/test_pipeline.py` | [x] | Each mode, evidence budget, thread-safe timings |
| `tests/test_api.py` | [x] | Endpoints, validation, request IDs, metrics, 502 on backend failure |
| `tests/test_llm.py` | [x] | HTTP payloads, API key, spec parsing |
| `tests/test_eval_harness.py` | [x] | End-to-end benchmark run, resume, reports |

### Infra / docs
| File | Status | Purpose |
|---|---|---|
| `pyproject.toml` | [x] | Project metadata, pytest + ruff config |
| `requirements.txt` | [x] | Full set (training + serving); old BLEU/ROUGE deps removed |
| `requirements-serve.txt` | [x] | API + benchmark, no GPU |
| `requirements-dev.txt` | [x] | Tests + lint |
| `Dockerfile` | [x] | CPU API image, non-root, healthcheck (builds in CI) |
| `docker-compose.yml` | [~] | Ollama + indexer + API stack (not run yet) |
| `.dockerignore` | [x] | |
| `.github/workflows/ci.yml` | [x] | Ruff + pytest on Python 3.10/3.12, Docker build (green on PR #1) |
| `LICENSE` | [x] | MIT; holder "a1if", year 2026 (check these) |
| `README.md` | [x] | Rewritten around the engineering story; results placeholder |
| `.gitignore` | [x] | Added `indexes/` |
| `notebooks/qlora_multihop_rag.ipynb` | [x] | Outputs and Colab widget state cleared |

---

## Phases

### Phase 1: Fix and restructure (done)
- [x] Fix multi-hop bugs (prompt echo, index metric mismatch, sampling, `<think>` leakage)
- [x] Model backend layer (in-process HF, or own checkpoint behind vLLM / Ollama)
- [x] FastAPI service with metrics and logging
- [x] HotpotQA benchmark harness
- [x] Tests, lint, CI, Docker, README
- [x] Commit and open PR: https://github.com/a1if/qlora-multihop-rag/pull/1
- [x] CI green on the PR (tests on 3.10 + 3.12, Docker build)
- [x] Clear notebook outputs
- [x] Reframe docs around fine-tuning (no hosted-API pitch)
- [ ] Check `LICENSE` holder/year

### Phase 2: Fine-tuning depth (the core story)
Compute: free Colab / Kaggle T4, or a rented GPU; QLoRA on 1.5B fits in 16 GB.

Infrastructure
- [ ] Config-driven training runs (YAML/dataclass per experiment, run name, seed)
- [ ] Experiment tracking (W&B or MLflow): loss curves, config, eval metrics per checkpoint
- [ ] Eval-on-checkpoint: every training run ends with the HotpotQA benchmark on a fixed dev slice
- [ ] Held-out split hygiene: train on HotpotQA *train*, report on *dev*, never tune on the final test slice

Baseline
- [ ] Base model, all four retrieval modes, n=500 (the "before" row)

Reader fine-tuning (RAFT-style SFT)
- [ ] Data builder: HotpotQA train question + gold passages + distractors → short answer with passage citations
- [ ] Rewrite `src/finetune.py` for this data (chat template, loss on the answer tokens only)
- [ ] First SFT run → benchmark `ft:single`, `ft:iterative` vs base

Ablations (what makes the fine-tune work)
- [ ] LoRA rank: 8 / 32 / 128
- [ ] Target modules: attention-only vs all projections
- [ ] QLoRA 4-bit vs 16-bit LoRA (quality vs memory)
- [ ] Training-set size: 1k / 5k / 20k examples (scaling curve)
- [ ] Distractor ratio in training contexts (robustness to irrelevant passages)
- [ ] With vs without reasoning traces in targets

Going deeper
- [ ] Fine-tune the follow-up query generator for iterative retrieval; measure support recall@k gain
- [ ] Preference tuning (DPO) on correct vs incorrect / unfaithful answers
- [ ] Error analysis: bridge vs comparison, failure categories, before/after examples

### Phase 3: Validate serving end-to-end
- [ ] Smoke run of the harness with a real model (`--n 20`)
- [ ] Fix whatever the first real run turns up (dataset loading, prompts, answer parsing)
- [ ] Serve the best checkpoint with vLLM or Ollama; API `curl /v1/answer` works
- [ ] `docker compose` stack up end-to-end (local Docker daemon currently returns 500; Ollama not installed)

### Phase 4: Write-up
- [ ] README results: baseline vs fine-tuned table, ablation tables, scaling plot
- [ ] Publish adapter to Hugging Face Hub with a model card
- [ ] Final resume bullets with real numbers

### Optional
- [ ] Hybrid retrieval (BM25 + dense) and cross-encoder reranker
- [ ] Faithfulness / citation-accuracy metric
- [ ] Streaming responses in the API
