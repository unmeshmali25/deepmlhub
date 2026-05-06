from dataclasses import dataclass


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    context_length: int = 1024
    n_positions: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    hidden_dim: int = 3072
    dropout: float = 0.0
    bias: bool = True

    @classmethod
    def gpt2_124m(cls) -> "GPT2Config":
        return cls(
            n_layer=12,
            n_head=12,
            n_embd=768,
            hidden_dim=3072,
        )

    @classmethod
    def gpt2_355m(cls) -> "GPT2Config":
        return cls(
            n_layer=24,
            n_head=16,
            n_embd=1024,
            hidden_dim=4096,
        )

    @classmethod
    def gpt2_774m(cls) -> "GPT2Config":
        return cls(
            n_layer=36,
            n_head=20,
            n_embd=1280,
            hidden_dim=5120,
        )

    @classmethod
    def gpt2_1_5b(cls) -> "GPT2Config":
        return cls(
            n_layer=48,
            n_head=25,
            n_embd=1600,
            hidden_dim=6400,
        )
