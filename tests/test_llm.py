import json

import pytest

httpx = pytest.importorskip("httpx")

from src.llm import OpenAICompatibleLLM, make_llm  # noqa: E402


def mock_client(captured):
    def handler(request):
        captured.append((request.url.path, json.loads(request.content), request.headers))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer: Poland"}}]})

    return httpx.Client(base_url="http://llm.test/v1", transport=httpx.MockTransport(handler))


def test_openai_backend_greedy_payload():
    captured = []
    llm = OpenAICompatibleLLM("qwen3:1.7b", "http://llm.test/v1", client=mock_client(captured))
    assert llm.generate("Q?", max_new_tokens=16) == "Answer: Poland"
    path, payload, _ = captured[0]
    assert path == "/v1/chat/completions"
    assert payload["model"] == "qwen3:1.7b"
    assert payload["max_tokens"] == 16
    assert payload["temperature"] == 0.0 and "top_p" not in payload


def test_openai_backend_sampling_payload():
    captured = []
    llm = OpenAICompatibleLLM("m", "http://llm.test/v1", client=mock_client(captured))
    llm.generate("Q?", max_new_tokens=8, do_sample=True, temperature=0.7, top_p=0.8)
    assert captured[0][1]["temperature"] == 0.7
    assert captured[0][1]["top_p"] == 0.8


def test_api_key_read_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = OpenAICompatibleLLM("m", "http://llm.test/v1")
    assert llm.client.headers["authorization"] == "Bearer sk-test"


def test_make_llm_parses_ollama_style_model_names():
    llm = make_llm("openai:qwen3:1.7b@http://localhost:11434/v1")
    assert llm.model == "qwen3:1.7b"
    assert str(llm.client.base_url).startswith("http://localhost:11434/v1")


@pytest.mark.parametrize("spec", ["nope", "hf:", "openai:no-url", "grpc:model@host"])
def test_make_llm_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        make_llm(spec)


class TemplateTokenizer:
    """Chat template without a thinking switch; vocab decides whether <think> exists."""

    def __init__(self, has_think_token):
        self.vocab = {"<think>": 1, "</think>": 2} if has_think_token else {"<|turn>": 1}

    def get_vocab(self):
        return self.vocab

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return f"<user>{messages[0]['content']}<model>\n"


def test_render_prompt_closes_think_block_for_r1_style_models():
    from src.llm import render_prompt

    prompt = render_prompt(TemplateTokenizer(has_think_token=True), "Q?", "chat", reasoning=False)
    assert prompt.endswith("<think>\n\n</think>\n\n")


def test_render_prompt_leaves_models_without_think_token_alone():
    from src.llm import render_prompt

    prompt = render_prompt(TemplateTokenizer(has_think_token=False), "Q?", "chat", reasoning=False)
    assert prompt == "<user>Q?<model>\n"
