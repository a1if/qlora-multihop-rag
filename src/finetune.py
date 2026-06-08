"""
src/finetune.py
QLoRA fine-tuning pipeline for DeepSeek-R1-Distill-Qwen-1.5B.

Usage:
    from src.finetune import run_finetuning
    from configs.training_config import FinetuneConfig

    run_finetuning(FinetuneConfig())
"""

import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# ─── Tokenizer & Model Loading ────────────────────────────────────────────────

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.unk_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name: str, config) -> AutoModelForCausalLM:
    """Load the quantised base model."""
    bnb_config = None
    if config.use_4bit:
        compute_dtype = (
            torch.float16 if config.bnb_4bit_compute_dtype == "float16" else torch.bfloat16
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_double_quant,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        max_memory={0: f"{config.max_memory_gb}GiB"},
        dtype=torch.float16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


# ─── Dataset Formatting ───────────────────────────────────────────────────────

def format_prompt(example: dict) -> dict:
    """
    Format a single Alpaca example into instruction-tuning prompt format.
    Adds guidance tokens to encourage lexically faithful responses.
    """
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()
    response = example["output"].strip()

    if input_text:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Guidance:\n"
            f"- Use the same key terms and phrasing found in the input.\n"
            f"- Keep the answer short, factual, and structured in sentences similar to the input.\n"
            f"- Avoid synonyms or rewording unless necessary.\n\n"
            f"### Response:\n{response}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Guidance:\n"
            f"- Keep the answer short, factual, and structured clearly.\n\n"
            f"### Response:\n{response}"
        )
    return {"text": text}


def prepare_dataset(config, tokenizer: AutoTokenizer):
    """Load, format, and tokenise the training dataset."""
    dataset = load_dataset(config.dataset_name, split=f"train[:{config.num_train_samples}]")
    formatted = dataset.map(format_prompt)
    tokenized = formatted.map(
        lambda ex: tokenizer(
            ex["text"],
            truncation=True,
            max_length=config.max_seq_length,
            add_special_tokens=True,
        ),
        batched=True,
        remove_columns=formatted.column_names,
    )
    return tokenized


# ─── Training ─────────────────────────────────────────────────────────────────

def build_lora_config(config) -> LoraConfig:
    """Construct the PEFT LoRA configuration."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
    )


def build_training_args(config) -> TrainingArguments:
    """Construct HuggingFace TrainingArguments from config."""
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        group_by_length=config.group_by_length,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to=config.report_to,
    )


# ─── Adapter Merging ──────────────────────────────────────────────────────────

def merge_and_save(adapter_path: str, output_dir: str):
    """
    Load the saved LoRA adapter, merge it into the base model weights,
    and save a standalone merged model in float32.
    """
    print("Merging LoRA adapter into base model...")
    peft_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    merged = peft_model.merge_and_unload()

    # Upcast to float32 for stable CPU inference
    for param in merged.parameters():
        param.data = param.data.float()

    merged.save_pretrained(output_dir)
    print(f"Merged model saved to: {output_dir}")


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def run_finetuning(config):
    """
    Full QLoRA fine-tuning pipeline:
        load model → prepare data → train → merge adapters → save
    """
    print(f"Loading base model: {config.base_model_name}")
    tokenizer = load_tokenizer(config.base_model_name)
    model = load_model(config.base_model_name, config)

    print("Preparing dataset...")
    dataset = prepare_dataset(config, tokenizer)

    lora_config = build_lora_config(config)
    training_args = build_training_args(config)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()
    print("Training complete.")

    # Save adapter
    trainer.model.save_pretrained(config.adapter_output_dir)
    tokenizer.save_pretrained(config.adapter_output_dir)

    # Merge and save final model
    merge_and_save(config.adapter_output_dir, config.merged_output_dir)
    tokenizer.save_pretrained(config.merged_output_dir)

    # Free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone. Fine-tuned model ready at: {config.merged_output_dir}")
