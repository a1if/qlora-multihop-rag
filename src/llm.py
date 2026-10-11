"""
src/llm.py
Pluggable LLM backends behind one small interface.

    hf:<path-or-hub-id>          local transformers model (e.g. the QLoRA merged model)
    openai:<model>@<base_url>    any OpenAI-compatible server: vLLM, Ollama, TGI, or a hosted API

The pipeline only ever calls `generate(...)`, so retrieval, evaluation and the
API service are identical whichever backend serves the model.

Usage:
    llm = make_llm("openai:deepseek-r1-1.5b@http://localhost:8000/v1")
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


# ─── Hugging Face (in-process) ────────────────────────────────────────────────

class HFLLM:
    """
    In-process transformers model.

    prompt_style:
        "chat"    the tokenizer's chat template (DeepSeek-R1-Distill native)
        "alpaca"  "### Instruction / ### Response" format used in finetune.py
    reasoning:
        False skips the R1 <think> phase by pre-filling an empty think block,
        which makes short-answer evaluation roughly 10x cheaper.
    """

    def __init__(self, model_path: str, prompt_style: str = "chat", reasoning: bool = False,
                 load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.name = f"hf:{model_path}"
        self._lock = threading.Lock()   # one generate() at a time on the shared model
        self.prompt_style = prompt_style
        self.reasoning = reasoning

        cuda = torch.cuda.is_available()
        quant = None
        if load_in_4bit and cuda:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if cuda else torch.float32,
            quantization_config=quant,
            device_map="auto" if cuda else None,
            trust_remote_code=True,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_prompt(self, instruction: str) -> str:
        if self.prompt_style == "alpaca":
            return f"### Instruction:\n{instruction}\n\n### Response:\n"

        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
        )
        if not self.reasoning:
            # Newer R1 templates already open the think block for us.
            prompt += "\n</think>\n\n" if prompt.rstrip().endswith("<think>") else "<think>\n\n</think>\n\n"
        return prompt

    def generate(self, instruction: str, max_new_tokens: int, do_sample: bool = False,
                 temperature: float = 0.3, top_p: float = 0.9, top_k: int = 40) -> str:
        import torch

        prompt = self.build_prompt(instruction)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            # the chat template already contains BOS
            add_special_tokens=self.prompt_style == "alpaca",
        ).to(self.model.device)

        sampling = dict(temperature=temperature, top_p=top_p, top_k=top_k) if do_sample else {}
        with self._lock, torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **sampling,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def close(self):
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─── OpenAI-compatible HTTP ───────────────────────────────────────────────────

class OpenAICompatibleLLM:
    """
    Calls /chat/completions on vLLM, Ollama (`/v1`), TGI or a hosted provider.
    The API key, if any, is read from OPENAI_API_KEY.
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
             load_in_4bit: bool = False) -> LLM:
    backend, sep, target = spec.partition(":")
    if not sep or not target:
        raise ValueError(f"Bad LLM spec {spec!r}; expected hf:<path> or openai:<model>@<base_url>")
    if backend == "hf":
        return HFLLM(target, prompt_style, reasoning, load_in_4bit)
    if backend == "openai":
        model, at, base_url = target.rpartition("@")
        if not at or not model:
            raise ValueError(f"Bad OpenAI spec {spec!r}; expected openai:<model>@<base_url>")
        return OpenAICompatibleLLM(model, base_url)
    raise ValueError(f"Unknown LLM backend {backend!r}; expected 'hf' or 'openai'")
