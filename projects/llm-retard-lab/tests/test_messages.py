import re
from llm_training.data.messages import to_message
from llm_training.data.messages import parse_pred
import json
import pytest
from llm_training.data.messages import load_jsonl


def test_to_message_basic():
    # set the input
    example = {"prompt": "hi", "response": "hello"}

    # call the function
    result = to_message(example)

    # assert
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == "hi"
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"] == "hello"


def test_parse_pred_with_thinking():
    # set the input
    text = "<thinking>Let me check the date.</thinking>\n\nRefund approved."

    # call the function
    result = parse_pred(text)

    assert result["thinking"] == "Let me check the date."
    assert result["reply"] == "Refund approved."


def test_parse_pred_no_thinking():
    result = parse_pred("  Just a plain answer.  ")

    assert result["thinking"] is None
    assert result["reply"] == "Just a plain answer."


def test_parse_pred_multiline_thinking():
    text = "<thinking>Line one\nLine two</thinking>\n\nReply here."
    result = parse_pred(text)

    assert result["thinking"] == "Line one\nLine two"
    assert result["reply"] == "Reply here."


def test_parse_pred_empty_thinking():
    result = parse_pred("<thinking></thinking>plain")

    assert result["thinking"] == ""
    assert result["reply"] == "plain"


def test_load_jsonl_basic(tmp_path):
    file = tmp_path / "data.jsonl"
    file.write_text('{"prompt": "a"}\n{"prompt": "b"}\n')

    result = load_jsonl(file)

    assert len(result) == 2
    assert result[0] == {"prompt": "a"}
    assert result[1] == {"prompt": "b"}


def test_load_jsonl_skips_blank_lines(tmp_path):
    file = tmp_path / "data.jsonl"
    file.write_text('{"prompt": "a"}\n\n{"prompt": "b"}\n')

    result = load_jsonl(file)

    assert len(result) == 2


def test_load_jsonl_invalid_json_raises(tmp_path):
    file = tmp_path / "data.jsonl"
    file.write_text('{"prompt": "a"}\nnot json\n')

    with pytest.raises(ValueError, match="data.jsonl:2"):
        load_jsonl(file)


def test_load_jsonl_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "nope.jsonl")
