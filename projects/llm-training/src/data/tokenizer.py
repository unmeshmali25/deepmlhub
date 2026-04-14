import tiktoken


class Tokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def decode_batch(self, tokens: list[list[int]]) -> list[str]:
        return [self.decode(t) for t in tokens]
