"""
src/text_utils.py
Parsing helpers for LLM outputs. Pure Python so they can be unit-tested
without loading a model.
"""

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
_ANSWER_LINE = re.compile(r"answer\**\s*:\s*(.+)", re.I)
_LIST_PREFIX = re.compile(
    r"^\s*(?:[-*•]|\(?\d+[.):]|sub-?questions?\s*\d*\s*[:.)]?|q\d+\s*[:.)])\s*", re.I
)


def strip_reasoning(text: str) -> str:
    """
    Remove DeepSeek-R1 style <think>...</think> reasoning.

    Handles three cases: a complete block, a block whose opening tag was part
    of the prompt (only `</think>` appears), and reasoning that was cut off
    by max_new_tokens (only `<think>` appears).
    """
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def extract_answer(text: str) -> str:
    """Pull a short answer out of a reader response ("Answer: X" preferred)."""
    text = strip_reasoning(text)
    matches = _ANSWER_LINE.findall(text)
    answer = matches[-1] if matches else _first_line(text)
    return answer.strip(" *\"'`.")


def extract_query(text: str) -> str:
    """Pull a single search query out of a follow-up generation."""
    line = _first_line(strip_reasoning(text))
    line = re.sub(r"^(search\s+)?query\s*:\s*", "", line, flags=re.I)
    return line.strip(" *\"'`")


def parse_subquestions(text: str, max_n: int) -> list[str]:
    """
    Parse a numbered / bulleted list of sub-questions.
    Lines ending in '?' are preferred; other lines are used only if none do.
    """
    lines = []
    for raw in strip_reasoning(text).splitlines():
        line = _LIST_PREFIX.sub("", raw).strip(" *\"'`")
        if line:
            lines.append(line)
    questions = [line for line in lines if line.endswith("?")]
    return (questions or lines)[:max_n]
