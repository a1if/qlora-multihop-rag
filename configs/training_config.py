"""
configs/training_config.py
Centralised hyperparameters for QLoRA fine-tuning.
Modify here rather than in training code.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class FinetuneConfig:
    # ── Model ────────────────────────────────────────────────────────────────
    base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    merged_output_dir: str = "./new_merged_model_fp32"
    adapter_output_dir: str = "./new_adapter"

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset_name: str = "yahma/alpaca-cleaned"
    num_train_samples: int = 2000
    max_seq_length: int = 512

    # ── Quantisation ─────────────────────────────────────────────────────────
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"             # nf4 | fp4
    bnb_4bit_compute_dtype: str = "float16"      # float16 | bfloat16
    use_double_quant: bool = False

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_r: int = 128
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # ── Training ─────────────────────────────────────────────────────────────
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4          # effective batch = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False                             # disabled for Turing GPUs (T4, 1660)
    gradient_checkpointing: bool = True
    group_by_length: bool = True

    # ── Logging ──────────────────────────────────────────────────────────────
    logging_steps: int = 25
    save_steps: int = 50
    report_to: str = "none"                        # "wandb" | "tensorboard" | "none"
    output_dir: str = "./results"

    # ── Memory ───────────────────────────────────────────────────────────────
    max_memory_gb: int = 6                         # target VRAM budget


@dataclass
class RAGConfig:
    # ── Knowledge Base ────────────────────────────────────────────────────────
    dataset_name: str = "gfissore/arxiv-abstracts-2021"
    num_documents: int = 5000
    index_path: str = "arxiv_abstracts.faiss"
    docs_path: str = "rag_documents"

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 8
    num_hops: int = 2
    max_context_chars: int = 2200

    # ── Generation ────────────────────────────────────────────────────────────
    max_new_tokens: int = 200
    temperature: float = 0.3
    top_p: float = 0.9
    top_k_sampling: int = 40
