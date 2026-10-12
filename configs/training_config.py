"""
configs/training_config.py
Centralised configuration for fine-tuning and the RAG pipeline.
Override per run with a JSON file or --set key=value (see src/finetune.py).
"""
import os
from dataclasses import dataclass, field



@dataclass
class FinetuneConfig:
    # ── Run ──────────────────────────────────────────────────────────────────
    run_name: str = "sft"
    output_root: str = field(default_factory=lambda: os.environ.get("QMR_RUNS_DIR", "runs"))
    seed: int = 0

    # ── Model ────────────────────────────────────────────────────────────────
    base_model_name: str = "Qwen/Qwen3-1.7B"
    method: str = "lora"                           # lora (bf16 base, fastest) | qlora (4-bit base, least memory)
    attn_implementation: str = "sdpa"              # sdpa | flash_attention_2 (if installed) | eager

    # ── Data (built by src/data_sft.py) ──────────────────────────────────────
    train_file: str = "data/sft/train.jsonl"
    val_file: str = "data/sft/val.jsonl"
    max_train_examples: int = 0                    # 0 = all; set for data-scaling runs
    tasks: list[str] = field(default_factory=lambda: ["reader", "abstain", "query"])
    max_seq_length: int = 1024                     # longer examples are dropped, never truncated
    max_val_examples: int = 300

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # ── Optimisation ─────────────────────────────────────────────────────────
    num_train_epochs: float = 1.0
    max_steps: int = -1                            # > 0 overrides epochs (throughput tests)
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2           # effective batch = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    bf16: bool = True                              # Ampere+ (RTX 30xx and newer); else fp16

    # ── Speed ────────────────────────────────────────────────────────────────
    gradient_checkpointing: bool = True            # off = faster, more memory
    group_by_length: bool = True                   # batch similar lengths, less padding
    optim: str = "adamw_torch_fused"               # fused AdamW kernel
    use_liger_kernel: bool = False                 # fused RMSNorm/SwiGLU/cross-entropy (needs liger-kernel + triton)
    allow_tf32: bool = True
    dataloader_num_workers: int = 0                # >0 is slow to start on Windows (spawn)

    # ── Logging / Saving ─────────────────────────────────────────────────────
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 500
    save_total_limit: int = 2
    report_to: str = "none"                        # none | tensorboard | wandb
    merge_after_training: bool = True

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_root, self.run_name)

    @property
    def adapter_dir(self) -> str:
        return os.path.join(self.run_dir, "adapter")

    @property
    def merged_dir(self) -> str:
        return os.path.join(self.run_dir, "merged")


@dataclass
class RAGConfig:
    # ── Knowledge Base (demo corpus) ──────────────────────────────────────────
    dataset_name: str = "gfissore/arxiv-abstracts-2021"
    num_documents: int = 5000
    index_dir: str = "indexes/arxiv"

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 5                                 # passages given to the reader (all modes)
    num_hops: int = 2
    max_passage_chars: int = 700

    # ── LLM Backend (see src/llm.py) ──────────────────────────────────────────
    # hf:<path-or-hub-id>  |  openai:<model>@<base_url>  (vLLM, Ollama, hosted APIs)
    llm: str = "hf:Qwen/Qwen3-1.7B"

    # ── Generation ────────────────────────────────────────────────────────────
    prompt_style: str = "chat"                     # chat | alpaca
    reasoning: bool = False                        # let the model emit <think> before answering
    reasoning_token_budget: int = 512              # extra tokens when reasoning=True
    load_in_4bit: bool = False                     # hf backend only
    max_new_tokens: int = 48                       # short answers (benchmarks)
    max_new_tokens_long: int = 256                 # free-form answers (demo)
    do_sample: bool = False                        # greedy decoding for reproducible evals
    temperature: float = 0.3
    top_p: float = 0.9
    top_k_sampling: int = 40
