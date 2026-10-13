"""
src/finetune.py
LoRA / QLoRA supervised fine-tuning of Qwen3-1.7B on the data from src/data_sft.py.

What it does:
    - renders each example with the same chat prompt used at inference
      (src.llm.render_prompt) and computes loss on the answer tokens only
    - drops over-length examples instead of truncating them
    - caches the tokenized dataset, so repeated runs skip tokenization
    - logs throughput (non-padding tokens/s) and peak GPU memory
    - saves the adapter, merges it into the base model for serving, and
      writes run_config.json + metrics.json into the run directory

Speed settings (see FinetuneConfig): bf16 LoRA instead of 4-bit QLoRA,
length-grouped batches, fused AdamW, TF32, optional gradient checkpointing
and Liger fused kernels.

Usage:
    python -m src.finetune                                         # defaults
    python -m src.finetune --set run_name=r8 --set lora_r=8        # override fields
    python -m src.finetune --config configs/runs/qlora.json        # JSON overrides
    python -m src.finetune --set max_steps=40 --set merge_after_training=false \\
        --set run_name=speed_test                                  # throughput test
"""

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time

from configs.training_config import FinetuneConfig

IGNORE_INDEX = -100


# ─── Config ───────────────────────────────────────────────────────────────────

def _parse_value(raw: str):
    """'32' → 32, 'false' → False, '["reader"]' → list; anything else stays a string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def load_config(path: str | None = None, overrides: list[str] = ()) -> FinetuneConfig:
    values = {}
    if path:
        with open(path, encoding="utf-8") as f:
            values.update(json.load(f))
    for item in overrides:
        key, eq, raw = item.partition("=")
        if not eq:
            raise ValueError(f"--set expects key=value, got {item!r}")
        values[key.strip()] = _parse_value(raw)
    known = {f.name for f in dataclasses.fields(FinetuneConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown FinetuneConfig fields: {unknown}")
    if values.get("method", "lora") not in ("lora", "qlora"):
        raise ValueError("method must be 'lora' or 'qlora'")
    return FinetuneConfig(**values)


def config_dict(config: FinetuneConfig) -> dict:
    return {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}


# ─── Tokenization ─────────────────────────────────────────────────────────────

def build_features(prompt_ids: list[int], completion_ids: list[int], end_id: int, max_len: int) -> dict | None:
    """Input ids and labels with the prompt masked out; None if too long."""
    input_ids = prompt_ids + completion_ids + [end_id]
    if len(input_ids) > max_len:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + completion_ids + [end_id],
        "length": len(input_ids),
    }


def end_of_turn_id(tokenizer) -> int:
    """Token that ends an assistant turn (<|im_end|> for Qwen chat models)."""
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
        return token_id
    return tokenizer.eos_token_id


def select_examples(examples, config: FinetuneConfig):
    import random

    chosen = [ex for ex in examples if ex.task in config.tasks]
    if config.max_train_examples:
        chosen = random.Random(config.seed).sample(chosen, min(config.max_train_examples, len(chosen)))
    return chosen


def tokenize(examples, tokenizer, max_len: int, batch_size: int = 512) -> tuple[list[dict], int]:
    from src.llm import render_prompt

    end_id = end_of_turn_id(tokenizer)
    features, dropped = [], 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        prompts = [ex.instruction if ex.prerendered else render_prompt(tokenizer, ex.instruction, "chat", reasoning=False)
                   for ex in batch]
        prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer([ex.response for ex in batch], add_special_tokens=False)["input_ids"]
        for p, c in zip(prompt_ids, completion_ids, strict=True):
            feature = build_features(p, c, end_id, max_len)
            if feature is None:
                dropped += 1
            else:
                features.append(feature)
    return features, dropped


def cache_key(config: FinetuneConfig, split: str, path: str) -> str:
    """Changes whenever the data file, tokenizer or filtering settings change."""
    stat = os.stat(path)
    parts = [split, path, str(stat.st_size), str(int(stat.st_mtime)), config.base_model_name,
             str(config.max_seq_length), ",".join(sorted(config.tasks)), str(config.max_train_examples),
             str(config.max_val_examples), str(config.seed)]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def load_split(config: FinetuneConfig, tokenizer, split: str):
    from datasets import Dataset, load_from_disk

    from src.data_sft import read_jsonl

    path = config.train_file if split == "train" else config.val_file
    cache_dir = os.path.join(config.output_root, "_tokenized", cache_key(config, split, path))
    if os.path.exists(cache_dir):
        return load_from_disk(cache_dir), None

    examples = read_jsonl(path)
    if split == "train":
        examples = select_examples(examples, config)
    else:
        examples = [ex for ex in examples if ex.task in config.tasks][:config.max_val_examples]
    features, dropped = tokenize(examples, tokenizer, config.max_seq_length)
    dataset = Dataset.from_list(features)
    dataset.save_to_disk(cache_dir)
    return dataset, dropped


# ─── Model ────────────────────────────────────────────────────────────────────

def load_model(config: FinetuneConfig):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = torch.bfloat16 if config.bf16 else torch.float16
    quant = None
    if config.method == "qlora":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name, dtype=dtype, quantization_config=quant,
        device_map={"": 0}, attn_implementation=config.attn_implementation,
    )
    model.config.use_cache = False

    if config.method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def merge_adapter(config: FinetuneConfig, tokenizer):
    """Fold the LoRA weights into a bf16 copy of the base model for serving."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(config.base_model_name, dtype=torch.bfloat16, device_map={"": 0})
    merged = PeftModel.from_pretrained(base, config.adapter_dir).merge_and_unload()
    merged.save_pretrained(config.merged_dir)
    tokenizer.save_pretrained(config.merged_dir)
    del base, merged
    torch.cuda.empty_cache()


# ─── Answer-span Loss ─────────────────────────────────────────────────────────

def answer_span_loss(model, inputs: dict, num_items_in_batch=None):
    """
    Causal-LM loss that runs the LM head only over the supervised (answer) span.

    Logits over a 152k-token vocabulary for every position dominate memory
    (8 x 1024 x 152k fp32 ≈ 5 GB), yet only the ~10-30 answer tokens per
    example carry loss. Positions are kept as one contiguous slice covering the
    answer tokens of every row in the batch, passed as `logits_to_keep`. The
    loss value is identical to the full computation.

    Returns (summed-loss / denominator, outputs, number of supervised tokens).
    """
    import torch
    import torch.nn.functional as F

    labels = inputs["labels"]
    # logits at position t predict the token at t + 1
    supervised = (labels[:, 1:] != IGNORE_INDEX).any(dim=0).nonzero().squeeze(-1)
    start, end = int(supervised.min()), int(supervised.max()) + 1
    keep = torch.arange(start, end, device=labels.device)

    model_inputs = {k: v for k, v in inputs.items() if k not in ("labels", "length")}
    outputs = model(**model_inputs, logits_to_keep=keep)
    logits = outputs.logits.float()
    targets = labels[:, start + 1:end + 1]
    loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=IGNORE_INDEX, reduction="sum")
    n_tokens = (targets != IGNORE_INDEX).sum()
    denominator = num_items_in_batch if num_items_in_batch is not None else n_tokens
    return loss_sum / denominator, outputs, n_tokens


def make_trainer_class():
    from transformers import Trainer

    class AnswerSpanTrainer(Trainer):
        # The loss below is normalised by the token count of the whole accumulated
        # batch (num_items_in_batch), so Trainer must not divide it again.
        loss_is_scaled_for_ga = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs, _ = answer_span_loss(model, inputs, num_items_in_batch)
            if num_items_in_batch is None and model.training:
                loss = loss / self.current_gradient_accumulation_steps
            return (loss, outputs) if return_outputs else loss

    return AnswerSpanTrainer


# ─── Training ─────────────────────────────────────────────────────────────────

def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_trainer(config: FinetuneConfig, model, tokenizer, train_ds, val_ds):
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir=os.path.join(config.run_dir, "checkpoints"),
        run_name=config.run_name,
        seed=config.seed,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_ratio,          # transformers 5: a float in [0, 1) is a ratio
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        bf16=config.bf16,
        fp16=not config.bf16,
        tf32=config.allow_tf32,
        use_liger_kernel=config.use_liger_kernel,
        train_sampling_strategy="group_by_length" if config.group_by_length else "random",
        length_column_name="length",
        dataloader_num_workers=config.dataloader_num_workers,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        include_num_input_tokens_seen="non_padding",
        remove_unused_columns=False,                # keep "length" for the length-grouped sampler
        prediction_loss_only=True,                  # never gather full-vocabulary logits in eval
    )
    pad = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=IGNORE_INDEX)

    def collator(features):
        # "length" is only for the sampler; the model must not receive it
        return pad([{k: v for k, v in f.items() if k != "length"} for f in features])

    trainer_cls = make_trainer_class() if config.answer_span_logits else Trainer
    return trainer_cls(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator)


def run_finetuning(config: FinetuneConfig) -> dict:
    import torch
    from transformers import AutoTokenizer, set_seed

    set_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    os.makedirs(config.run_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, train_dropped = load_split(config, tokenizer, "train")
    val_ds, _ = load_split(config, tokenizer, "val") if config.val_file and config.max_val_examples else (None, None)
    print(f"train examples: {len(train_ds)}" + (f" (dropped {train_dropped} over {config.max_seq_length} tokens)"
                                                 if train_dropped else ""))

    with open(os.path.join(config.run_dir, "run_config.json"), "w") as f:
        json.dump({"config": config_dict(config), "git_commit": _git_commit(),
                   "train_examples": len(train_ds), "val_examples": len(val_ds) if val_ds else 0}, f, indent=2)

    model = load_model(config)
    model.print_trainable_parameters()
    trainer = build_trainer(config, model, tokenizer, train_ds, val_ds)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start

    metrics = {
        "train_loss": result.training_loss,
        "train_runtime_s": elapsed,
        "steps": trainer.state.global_step,
        "tokens_seen": trainer.state.num_input_tokens_seen,
        "tokens_per_s": trainer.state.num_input_tokens_seen / elapsed if elapsed else 0.0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
    }
    if val_ds is not None:
        metrics["eval_loss"] = trainer.evaluate()["eval_loss"]

    model.save_pretrained(config.adapter_dir)
    tokenizer.save_pretrained(config.adapter_dir)
    del trainer, model
    torch.cuda.empty_cache()

    if config.merge_after_training:
        merge_adapter(config, tokenizer)

    with open(os.path.join(config.run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    if config.merge_after_training:
        print(f"\nBenchmark it with:\n  python -m src.eval_hotpotqa --model ft=hf:{config.merged_dir} "
              f"--systems ft:single ft:iterative")
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file of FinetuneConfig overrides")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)
    run_finetuning(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
