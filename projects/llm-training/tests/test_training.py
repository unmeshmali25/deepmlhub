import pytest
import torch

from src.data.tokenizer import Tokenizer


class TestTokenizer:
    def test_encode_decode(self):
        tokenizer = Tokenizer("gpt2")
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert text == decoded

    def test_encode_batch(self):
        tokenizer = Tokenizer("gpt2")
        texts = ["Hello", "World", "Test"]
        batch = tokenizer.encode_batch(texts)
        assert len(batch) == 3
        assert all(isinstance(t, list) for t in batch)

    def test_vocab_size(self):
        tokenizer = Tokenizer("gpt2")
        assert tokenizer.vocab_size == 50257
