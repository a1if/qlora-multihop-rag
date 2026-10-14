"""Training-side logic that doesn't need a GPU: config, label masking, prompt parity."""

import pytest

from src.data_sft import SFTExample
from src.finetune import IGNORE_INDEX, build_features, cache_key, load_config, select_examples, tokenize
from src.llm import render_prompt


class FakeTokenizer:
    """Whitespace tokenizer with a Qwen-style chat template."""

    unk_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.vocab = {"<unk>": 0, "<eos>": 1, "<|im_end|>": 2}

    def _id(self, word):
        return self.vocab.setdefault(word, len(self.vocab))

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        prompt = f"<user> {messages[0]['content']} <assistant>"
        return prompt + (" <think> </think> " if not enable_thinking else " ")

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[self._id(w) for w in t.split()] for t in texts]}


def test_load_config_overrides_and_types(tmp_path):
    path = tmp_path / "run.json"
    path.write_text('{"lora_r": 8, "method": "qlora"}')
    config = load_config(str(path), ["run_name=r8", "gradient_checkpointing=false", 'tasks=["reader"]',
                                     "learning_rate=1e-4"])
    assert (config.lora_r, config.method, config.run_name) == (8, "qlora", "r8")
    assert config.gradient_checkpointing is False
    assert config.tasks == ["reader"]
    assert config.learning_rate == 1e-4
    assert config.merged_dir.endswith("r8/merged") or config.merged_dir.endswith("r8\\merged")


@pytest.mark.parametrize("override", ["lora_rank=8", "nokey", "method=full"])
def test_load_config_rejects_bad_input(override):
    with pytest.raises(ValueError):
        load_config(None, [override])


def test_build_features_masks_prompt():
    f = build_features([10, 11, 12], [20, 21], end_id=2, max_len=16)
    assert f["input_ids"] == [10, 11, 12, 20, 21, 2]
    assert f["labels"] == [IGNORE_INDEX] * 3 + [20, 21, 2]
    assert f["length"] == 6 and f["attention_mask"] == [1] * 6


def test_build_features_drops_over_length():
    assert build_features([1] * 10, [2] * 5, end_id=2, max_len=15) is None


def test_tokenize_uses_inference_prompt_and_end_of_turn():
    tok = FakeTokenizer()
    ex = SFTExample("q1", "reader", "What is two plus two ?", "Answer: four")
    features, dropped = tokenize([ex], tok, max_len=64)
    assert dropped == 0
    prompt_len = len(render_prompt(tok, ex.instruction, "chat", reasoning=False).split())
    labels = features[0]["labels"]
    assert labels[:prompt_len] == [IGNORE_INDEX] * prompt_len       # no loss on the prompt
    assert labels[-1] == tok.vocab["<|im_end|>"]                    # learns to stop
    assert len(labels) == prompt_len + 2 + 1


def test_tokenize_counts_dropped():
    tok = FakeTokenizer()
    long = SFTExample("q2", "reader", "word " * 100, "Answer: x")
    _, dropped = tokenize([long], tok, max_len=32)
    assert dropped == 1


def test_select_examples_filters_and_subsamples():
    examples = [SFTExample(str(i), "query" if i % 3 == 0 else "reader", "i", "r") for i in range(30)]
    config = load_config(None, ['tasks=["reader"]', "max_train_examples=5"])
    chosen = select_examples(examples, config)
    assert len(chosen) == 5 and all(ex.task == "reader" for ex in chosen)
    assert chosen == select_examples(examples, config)               # deterministic


def test_cache_key_changes_with_settings(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n")
    a = cache_key(load_config(None, []), "train", str(data))
    b = cache_key(load_config(None, ["max_seq_length=512"]), "train", str(data))
    assert a != b
    assert a == cache_key(load_config(None, ["lora_r=8"]), "train", str(data))  # LoRA rank doesn't affect data
