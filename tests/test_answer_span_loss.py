"""
The answer-span loss must equal the standard causal-LM loss while computing
logits for far fewer positions. Runs a tiny random Qwen3 on CPU; skipped when
torch / transformers aren't installed (the lightweight CI environment).
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.finetune import IGNORE_INDEX, answer_span_loss  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    config = transformers.Qwen3Config(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(config).eval()


def make_batch(prompt_lens, answer_lens, total, seed=0):
    """Right-padded batch where only the answer tokens carry labels."""
    g = torch.Generator().manual_seed(seed)
    rows = len(prompt_lens)
    input_ids = torch.zeros(rows, total, dtype=torch.long)
    attention = torch.zeros(rows, total, dtype=torch.long)
    labels = torch.full((rows, total), IGNORE_INDEX)
    for i, (p, a) in enumerate(zip(prompt_lens, answer_lens, strict=True)):
        ids = torch.randint(1, 512, (p + a,), generator=g)
        input_ids[i, :p + a] = ids
        attention[i, :p + a] = 1
        labels[i, p:p + a] = ids[p:]
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


@pytest.mark.parametrize("prompt_lens,answer_lens", [
    ([30, 30], [5, 5]),          # same layout in every row
    ([40, 22, 31], [3, 9, 6]),   # different prompt and answer lengths
])
def test_matches_full_loss(tiny_model, prompt_lens, answer_lens):
    batch = make_batch(prompt_lens, answer_lens, total=64)
    with torch.no_grad():
        full = tiny_model(**batch).loss                     # HF mean over supervised tokens
        span, outputs, n_tokens = answer_span_loss(tiny_model, batch)
    assert span.item() == pytest.approx(full.item(), rel=1e-5)
    assert n_tokens.item() == sum(answer_lens)
    # Only the answer span got logits, not all 64 positions.
    assert outputs.logits.shape[1] < 64 / 2


def test_normalises_by_accumulated_token_count(tiny_model):
    batch = make_batch([30, 30], [5, 5], total=48)
    with torch.no_grad():
        per_batch, _, n = answer_span_loss(tiny_model, batch)
        accumulated, _, _ = answer_span_loss(tiny_model, batch, num_items_in_batch=4 * n)
    assert accumulated.item() == pytest.approx(per_batch.item() / 4, rel=1e-6)


def test_gradients_match_full_loss(tiny_model):
    batch = make_batch([20, 26], [4, 7], total=40)
    tiny_model.zero_grad()
    tiny_model(**batch).loss.backward()
    full_grads = [p.grad.clone() for p in tiny_model.parameters()]
    tiny_model.zero_grad()
    answer_span_loss(tiny_model, batch)[0].backward()
    for full, span in zip(full_grads, (p.grad for p in tiny_model.parameters()), strict=True):
        assert torch.allclose(full, span, atol=1e-6)
