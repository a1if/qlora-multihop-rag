# Task Tracker: Resume Upgrade

Branch: `eval-hotpotqa` · Status: uncommitted · Tests: 57 passing · Lint: clean

Goal: turn the coursework repo into an engineering-focused project (serving, evaluation, testing, CI) with a measured headline result.

Legend: `[x]` done · `[ ]` todo · `[~]` written but not checked against real models/data

---

## Files

### Source: new
| File | Status | Purpose |
|---|---|---|
| `src/llm.py` | [~] | `LLM` protocol; `HFLLM` (local transformers) and `OpenAICompatibleLLM` (vLLM / Ollama / hosted); `make_llm(spec)` |
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
| `src/finetune.py` | [ ] | Unchanged; still Alpaca training (see Phase 3) |
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
| `Dockerfile` | [~] | CPU API image, non-root, healthcheck (not built yet) |
| `docker-compose.yml` | [~] | Ollama + indexer + API stack (not run yet) |
| `.dockerignore` | [x] | |
| `.github/workflows/ci.yml` | [~] | Ruff + pytest on Python 3.10/3.12, Docker build (runs after first push) |
| `LICENSE` | [x] | MIT; holder "a1if", year 2026 (check these) |
| `README.md` | [x] | Rewritten around the engineering story; results placeholder |
| `.gitignore` | [x] | Added `indexes/` |
| `notebooks/qlora_multihop_rag.ipynb` | [ ] | Still shows old BLEU 0.000 outputs |

---

## Phases

### Phase 1: Fix and restructure (done)
- [x] Fix multi-hop bugs (prompt echo, index metric mismatch, sampling, `<think>` leakage)
- [x] Pluggable LLM backends
- [x] FastAPI service with metrics and logging
- [x] HotpotQA benchmark harness
- [x] Tests, lint, CI, Docker, README

### Phase 2: Validate and get the headline number
- [ ] Commit Phase 1 and push the branch; confirm CI passes
- [ ] Clear outputs in the notebook (or remove it)
- [ ] Check `LICENSE` holder/year
- [ ] Smoke run: `python -m src.eval_hotpotqa --n 20 --systems base:closed base:single` against Ollama or vLLM
- [ ] Fix whatever the first real run turns up (dataset loading, prompts, answer parsing)
- [ ] `docker compose` stack up end-to-end; `curl /v1/answer` works
- [ ] Full run: `--n 500`, all four modes
- [ ] Paste `results/hotpotqa/summary.md` into README
- [ ] Write final resume bullets with real numbers

### Phase 3: Make the fine-tune earn its place
- [ ] RAFT-style data: HotpotQA train questions + gold and distractor passages → short cited answer
- [ ] Update `src/finetune.py` to use it (chat template, not Alpaca)
- [ ] Serve the merged model via vLLM; benchmark `ft:*` vs `base:*`

### Phase 4: Optional extras
- [ ] Hybrid retrieval (BM25 + dense) and cross-encoder reranker; report recall@k gain
- [ ] Faithfulness / citation-accuracy metric
- [ ] Publish adapter to Hugging Face Hub with a model card; demo on HF Spaces
- [ ] Streaming responses in the API
