# Task Tracker: Resume Upgrade

Branch: `eval-hotpotqa` · PR #1 open, CI green · Tests: 57 passing · Lint: clean

Goal: take **one** small open model (Qwen3-1.7B) and improve it step by step for multi-hop QA with RAG and fine-tuning, measuring every step on HotpotQA. No model comparisons, no cloud models. The resume story is the skills: RAG design, fine-tuning (QLoRA/LoRA, retriever tuning), evaluation, and shipping it as a service.

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
| `scripts/setup_gpu_env.ps1` | [x] | Windows GPU env: venv + HF cache on another drive, PyTorch CUDA 12.8 |
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

### Phase 2: Improvement ladder (the core story)
One model, Qwen3-1.7B. Each stage adds one technique on top of the previous one and gets a row in the results table (EM, F1, retrieval recall@k, latency).
Compute: local RTX 5070 (12 GB), models straight from Hugging Face (`hf:` backend). Needs PyTorch built for CUDA 12.8+; bf16 supported.

Setup
- [x] Model: **Qwen3-1.7B** (thinking toggled via `enable_thinking`; bf16)
- [x] Local GPU environment at `D:\qmr` (C: is full): torch 2.11+cu128, transformers 5.19, peft 0.21, trl 1.15, bitsandbytes 0.50 (`scripts/setup_gpu_env.ps1`)
- [x] Smoke run on the real model (`--n 20`): pipeline works end to end. Fixed `torch_dtype` → `dtype` for transformers 5
  - finding: base model's hop-2 queries changed retrieval on 12/20 questions but found 0 extra gold passages, which motivates Stages 4–5
- [ ] Full baseline n=500 for Stages 0–2 (running → `results/baseline/summary.md`)
- [ ] Split hygiene: train on HotpotQA *train*, report on a fixed *dev* sample (n=500)
- [ ] Experiment tracking (W&B or MLflow): config, loss curves, eval per run

The ladder
- [ ] Stage 0: no retrieval (closed-book baseline)
- [ ] Stage 1: + RAG, single-hop dense retrieval
- [ ] Stage 2: + multi-hop retrieval (iterative follow-up search; decompose as a comparison)
- [ ] Stage 3: + better retrieval: hybrid BM25 + dense, cross-encoder reranker
- [ ] Stage 4: + fine-tuned retriever (contrastive tuning of the embedder on question → gold passages)
- [ ] Stage 5: + fine-tuned model, QLoRA SFT, one LoRA adapter trained on two tasks:
  - reader: question + gold and distractor passages → short answer with citations, "insufficient evidence" when gold is missing
  - query writer: question + hop-1 passages → hop-2 search query (bridge entity)
- [ ] Stage 6 (optional): + DPO on grounded vs unsupported answers sampled from the Stage 5 model

Code needed for the ladder
- [ ] Data builder for the SFT tasks (`src/data_sft.py`)
- [ ] Rewrite `src/finetune.py`: chat template, loss on answer tokens only, bf16, config per run
- [ ] Retriever fine-tuning script (`src/finetune_retriever.py`)
- [ ] BM25 + reranker in `src/retriever.py`
- [ ] Harness: select retriever variant per stage (`--retriever dense|hybrid|tuned`)
- [ ] Extra metrics: citation precision/recall, hallucination rate on unanswerable variants

Understanding why it works (small ablations on Stage 5)
- [ ] LoRA rank 8 vs 32 vs 128
- [ ] QLoRA 4-bit vs 16-bit LoRA (quality vs memory)
- [ ] Training-set size 1k / 5k / 20k
- [ ] Error analysis: bridge vs comparison questions, before/after examples

### Phase 3: Ship the final model
- [ ] Serve the best checkpoint with vLLM or Ollama; API `curl /v1/answer` works
- [ ] Optional: `docker compose` stack (Docker daemon currently returns 500). Not needed for training or benchmarking.

### Phase 4: Write-up
- [ ] README results: ladder table (Stage 0 → final), ablation table, data-size plot
- [ ] README "What I learned" section
- [ ] Publish adapter to Hugging Face Hub with a model card
- [ ] Final resume bullets with real numbers

### Optional
- [ ] Streaming responses in the API
