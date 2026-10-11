from src.text_utils import extract_answer, extract_query, parse_subquestions, strip_reasoning


def test_strip_reasoning_complete_block():
    assert strip_reasoning("<think>hmm let me see</think>\nAnswer: Paris") == "Answer: Paris"


def test_strip_reasoning_open_tag_in_prompt():
    # The chat template opened <think>, so only the closing tag is generated.
    assert strip_reasoning("reasoning here\n</think>\n\nAnswer: Paris") == "Answer: Paris"


def test_strip_reasoning_truncated():
    assert strip_reasoning("Answer: Paris <think>never closed") == "Answer: Paris"


def test_extract_answer_prefers_last_answer_line():
    text = "The passage says...\nAnswer: Warsaw\nWait, re-checking.\nAnswer: Poland"
    assert extract_answer(text) == "Poland"


def test_extract_answer_markdown_and_punctuation():
    assert extract_answer("**Answer:** \"Marie Curie\".") == "Marie Curie"


def test_extract_answer_falls_back_to_first_line():
    assert extract_answer("\n\nyes\nbecause both are...") == "yes"


def test_extract_query():
    assert extract_query("<think>x</think>Search query: \"Marie Curie birthplace\"") == "Marie Curie birthplace"


def test_parse_subquestions_numbered_and_bulleted():
    text = "Sub-questions:\n1. Who discovered polonium?\n2) Where was she born?\n- What country is that in?"
    assert parse_subquestions(text, 2) == ["Who discovered polonium?", "Where was she born?"]


def test_parse_subquestions_without_question_marks():
    assert parse_subquestions("1. find the discoverer\n2. find birthplace", 3) == ["find the discoverer", "find birthplace"]
