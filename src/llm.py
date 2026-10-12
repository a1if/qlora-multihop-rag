"""
src/llm.py
Pluggable LLM backends behind one small interface.

    hf:<path-or-hub-id>          local transformers model (e.g. the QLoRA merged model)
    openai:<model>@<base_url>    a checkpoint served by vLLM, Ollama or TGI

The pipeline only ever calls `generate(...)`, so the base model and every
fine-tuned checkpoint go through identical retrieval, evaluation and serving,
whether loaded in-process or behind a production inference server.

Usage:
    llm = make_llm("openai:qwen3-1.7b@http://localhost:8000/v1")
    llm = make_llm("hf:./new_merged_model_fp32", prompt_style="alpaca")
"""

import gc
import os
import threading
from typing import Protocol


class LLM(Protocol):
    name: str

    def generate(self, instruction: str, max_new_tokens: int, do_sample: bool = False,
                 temperature: float = 0.3, top_p: float = 0.9, top_k: int = 40) -> str:
        """Return only the newly generated text for a single-turn instruction."""

    def close(self) -> None:
        ...


# ─── Prompt Rendering ─────────────────────────────────────────────────────────

def render_prompt(tokenizer, instruction: str, prompt_style: str = "chat", reasoning: bool = False) -> str:
    """
    Text fed to the model for one user instruction, up to where the answer starts.
    Shared by inference (HFLLM) and fine-tuning (src/finetune.py) so both see
    exactly the same format.
    """
    if prompt_style == "alpaca":
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True,
        enable_thinking=reasoning,   # Qwen3 switch; ignored by other templates
    )
    if not reasoning and "</think>" not in prompt[-64:]:
        # Template has no thinking switch (R1-Distill): close the think block ourselves.
        prompt += "\n</think>\n\n" if prompt.rstrip().endswith("<think>") else "<think>\n\n</think>\n\n"
    return prompt


# ─── Hugging Face (in-process) ────────────────────────────────────────────────

class HFLLM:
    """
    In-process transformers model.

    prompt_style:
        "chat"    the tokenizer's chat template (Qwen3 / R1-Distill native)
        "alpaca"  plain "### Instruction / ### Response" format
    reasoning:
        False turns off the <think> phase: Qwen3 templates do this via
        `enable_thinking=False`; for templates without that switch (R1-Distill)
        an empty think block is pre-filled. Short answers get ~10x cheaper.
    max_batch_size:
        > 1 enables dynamic batching (src/batching.py): concurrent generate()
        calls from different threads are run as one left-padded GPU batch.
        Note: batched greedy decoding can differ from unbatched in rare cases
        because padded matmuls round differently in bf16.
    """

    def __init__(self, model_path: str, prompt_style: str = "chat", reasoning: bool = False,
                 load_in_4bit: bool = False, max_batch_size: int = 1, batch_wait_ms: float = 10.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.name = f"hf:{model_path}"
        self._lock = threading.Lock()   # one generate() at a time on the shared model
        self.prompt_style = prompt_style
        self.reasoning = reasoning

        cuda = torch.cuda.is_available()
        dtype = torch.float32
        if cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant = None
        if load_in_4bit and cuda:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            quantization_config=quant,
            device_map="auto" if cuda else None,
            trust_remote_code=True,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"    # generation continues from the right edge
        # generation_config may list several stop tokens (Qwen3: <|im_end|> and <|endoftext|>)
        self.eos_ids = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id

        self._batcher = None
        if max_batch_size > 1:
            from src.batching import DynamicBatcher

            self._batcher = DynamicBatcher(self._generate_batch, max_batch_size, batch_wait_ms)

    @property
    def batch_sizes(self) -> list[int]:
        return self._batcher.batch_sizes if self._batcher else []

    def build_prompt(self, instruction: str) -> str:
        return render_prompt(self.tokenizer, instruction, self.prompt_style, self.reasoning)

    def generate(self, instruction: str, max_new_tokens: int, do_sample: bool = False,
                 temperature: float = 0.3, top_p: float = 0.9, top_k: int = 40) -> str:
        sampling = (("temperature", temperature), ("top_p", top_p), ("top_k", top_k)) if do_sample else ()
        payload = (self.build_prompt(instruction), max_new_tokens, sampling)
        if self._batcher is None:
            with self._lock:
                return self._generate_batch([payload])[0]
        # Only requests with the same decoding settings can share a generate() call.
        return self._batcher.submit(payload, group=sampling)

    def _generate_batch(self, payloads: list[tuple]) -> list[str]:
        import torch

        prompts = [p[0] for p in payloads]
        limits = [p[1] for p in payloads]
        sampling = dict(payloads[0][2])
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            # the chat template already contains BOS
            add_special_tokens=self.prompt_style == "alpaca",
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max(limits),
                do_sample=bool(sampling),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_ids,
                **sampling,
            )
        start = inputs["input_ids"].shape[1]
        # Each caller gets at most its own max_new_tokens, even if the batch ran longer.
        return [self.tokenizer.decode(output[i, start:start + limit], skip_special_tokens=True)
                for i, limit in enumerate(limits)]

    def close(self):
        import torch

        if self._batcher is not None:
            self._batcher.close()
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─── OpenAI-compatible HTTP ───────────────────────────────────────────────────

class OpenAICompatibleLLM:
    """
    Calls /chat/completions on an OpenAI-compatible inference server (vLLM,
    Ollama `/v1`, TGI). An API key, if the server needs one, is read from OPENAI_API_KEY.
    """

    def __init__(self, model: str, base_url: str, timeout_s: float = 120.0, max_retries: int = 3,
                 client=None):
        import httpx

        self.name = f"openai:{model}@{base_url}"
        self.model = model
        headers = {}
        if os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
        self.client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_s,
            transport=httpx.HTTPTransport(retries=max_retries),
        )

    def generate(self, instruction: str, max_new_tokens: int, do_sample: bool = False,
                 temperature: float = 0.3, top_p: float = 0.9, top_k: int = 40) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": instruction}],
            "max_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 0.0,
        }
        if do_sample:
            payload["top_p"] = top_p
        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        return message.get("content") or ""

    def close(self):
        self.client.close()


# ─── Factory ──────────────────────────────────────────────────────────────────

def make_llm(spec: str, prompt_style: str = "chat", reasoning: bool = False,
             load_in_4bit: bool = False, max_batch_size: int = 1) -> LLM:
    backend, sep, target = spec.partition(":")
    if not sep or not target:
        raise ValueError(f"Bad LLM spec {spec!r}; expected hf:<path> or openai:<model>@<base_url>")
    if backend == "hf":
        return HFLLM(target, prompt_style, reasoning, load_in_4bit, max_batch_size)
    if backend == "openai":
        model, at, base_url = target.rpartition("@")
        if not at or not model:
            raise ValueError(f"Bad OpenAI spec {spec!r}; expected openai:<model>@<base_url>")
        return OpenAICompatibleLLM(model, base_url)
    raise ValueError(f"Unknown LLM backend {backend!r}; expected 'hf' or 'openai'")
