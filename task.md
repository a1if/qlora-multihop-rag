# Task Tracker: Resume Upgrade

Branch: `eval-hotpotqa` · PR #1 open, CI green · Tests: 81 passing · Lint: clean

Goal: take **one** small open model (Qwen3-1.7B) and improve it step by step for multi-hop QA with RAG and fine-tuning, measuring every step on HotpotQA. No model comparisons, no cloud models. The resume story is the skills: RAG design, fine-tuning (QLoRA/LoRA, retriever tuning), evaluation, and shipping it as a service.

Legend: `[x]` done · `[ ]` todo · `[~]` written but not checked against real models/data

---

## Resume here (paused 2026-10-12)

Baseline results so far (Qwen3-1.7B, HotpotQA dev n=500, k=5):

| Stage | EM | F1 (95% CI) | Support R@5 | All-support@5 | p50 latency |
|---|---|---|---|---|---|
| 0 closed-book | 10.4 | 13.9 [11.2, 16.8] | – | – | 0.38 s |
| 1 single-hop RAG | 34.0 | 43.2 [39.1, 47.4] | 75.3 | 54.4 | 0.47 s |
| 2 iterative multi-hop | 37.2 | 47.8 [43.6, 52.0] | 77.8 | 59.8 | 1.28 s |

Next steps, in order:
1. Finish the baseline (decompose resumes from 231/500; now batched, so much faster):
   `python -m src.eval_hotpotqa --n 500 --systems base:closed base:single base:iterative base:decompose --reference base:single --out results/baseline`
2. Batching check: rerun `base:single base:iterative` with `--out results/batched`; compare wall time and answers with `results/baseline`
3. Training throughput sweep: `python -m src.finetune --set max_steps=40 --set merge_after_training=false --set run_name=speed_<variant> ...` over lora/qlora × gradient checkpointing on/off × batch size
4. First full SFT run, then benchmark `ft:single ft:iterative`

Environment (every new shell):
```powershell
$env:HF_HOME = "D:\qmr\hf-cache"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:QMR_RUNS_DIR = "D:\qmr\runs"
& D:\qmr\venv\Scripts\Activate.ps1
cd C:\Users\alift\qlora-multihop-rag
```
Local, gitignored artifacts: `results/baseline/` (predictions + summary) and `data/sft/` (29,715 train / 743 val). Regenerate the data with `python -m src.data_sft --n 20000 --out data/sft`.

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
| `src/batching.py` | [~] | Dynamic batching of concurrent generate() calls (tested with fakes; GPU check pending) |
| `src/build_index.py` | [~] | Index build CLI (arXiv or HotpotQA) |
| `src/data_sft.py` | [x] | SFT data from HotpotQA train: reader (gold + hard distractors → answer + citations), abstain, hop-2 query writer |

### Source: modified
| File | Status | Change |
|---|---|---|
| `src/rag_pipeline.py` | [~] | Rewritten: 4 modes (closed / single / iterative / decompose), same passage budget for all, per-thread stage timings; fixed prompt-in-output, L2/cosine mismatch and sampling bugs |
| `src/app.py` | [~] | Gradio demo on the new pipeline, configured from env vars |
| `configs/training_config.py` | [x] | `RAGConfig` gains `llm` spec, prompt style, reasoning, generation settings; GPU-specific comments removed |
| `src/finetune.py` | [~] | Rewritten for the SFT data: LoRA/QLoRA, answer-only loss, speed settings, merge for serving (not yet run on GPU) |
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
| `tests/test_batching.py` | [x] | Batching, max size, group isolation, error propagation |
| `tests/test_finetune.py` | [x] | Config overrides, label masking, prompt parity with inference, end-of-turn token, data cache key |
| `tests/test_data_sft.py` | [x] | Task mix, gold/citation correctness, abstain support, bridge targets, train/inference prompt parity |

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
- [~] Full baseline n=500 for Stages 0–2: closed, single, iterative done; decompose paused at 231/500 (resumes automatically)
- [ ] Split hygiene: train on HotpotQA *train*, report on a fixed *dev* sample (n=500)
- [ ] Experiment tracking (W&B or MLflow): config, loss curves, eval per run

The ladder
- [x] Stage 0: no retrieval (closed-book baseline): EM 10.4 · F1 13.9
- [x] Stage 1: + RAG, single-hop dense retrieval: EM 34.0 · F1 43.2 · support R@5 75.3%
- [~] Stage 2: + multi-hop retrieval: iterative EM 37.2 · F1 47.8 (+4.6 F1 vs single, 95% CI [+1.7, +7.5]) · support R@5 77.8%; decompose pending
- [ ] Stage 3: + better retrieval: hybrid BM25 + dense, cross-encoder reranker
- [ ] Stage 4: + fine-tuned retriever (contrastive tuning of the embedder on question → gold passages)
- [ ] Stage 5: + fine-tuned model, QLoRA SFT, one LoRA adapter trained on two tasks:
  - reader: question + gold and distractor passages → short answer with citations, "insufficient evidence" when gold is missing
  - query writer: question + hop-1 passages → hop-2 search query (bridge entity)
- [ ] Stage 6 (optional): + DPO on grounded vs unsupported answers sampled from the Stage 5 model

Code needed for the ladder
- [x] Data builder for the SFT tasks (`src/data_sft.py`): 20k train questions → 29,715 examples (17,847 reader · 2,031 abstain · 9,837 query), 743 val; prompts shared with the pipeline so train = inference text
- [x] Rewrite `src/finetune.py`: same chat prompt as inference, loss on answer tokens only, over-length examples dropped (never truncated), tokenized-data cache, JSON/`--set` config per run, adapter + merged model + run_config.json + metrics.json (tokens/s, peak VRAM)
- [ ] Throughput sweep on the 5070 (after baseline finishes): LoRA vs QLoRA, gradient checkpointing on/off, batch size → pick fastest setting that fits; record speed-up
- [ ] Retriever fine-tuning script (`src/finetune_retriever.py`)
- [ ] BM25 + reranker in `src/retriever.py`
- [ ] Harness: select retriever variant per stage (`--retriever dense|hybrid|tuned`)
- [ ] Extra metrics: citation precision/recall, hallucination rate on unanswerable variants

Speed
- [x] Dynamic batching for local-model evaluation and the API (`src/batching.py`, `--workers 16` default)
- [ ] Verify batching on GPU: wall time vs unbatched baseline, and answer agreement on the same 500 questions
- [ ] Training throughput sweep (see above)

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
