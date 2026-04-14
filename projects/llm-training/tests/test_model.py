import torch

from src.models.gpt2.config import GPT2Config
from src.models.gpt2.model import GPT2Model


class TestGPT2Model:
    def test_gpt2_124m_forward(self):
        config = GPT2Config.gpt2_124m()
        model = GPT2Model.from_config(config)

        batch_size = 2
        seq_len = 16
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(x)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is None

    def test_gpt2_124m_with_targets(self):
        config = GPT2Config.gpt2_124m()
        model = GPT2Model.from_config(config)

        batch_size = 2
        seq_len = 16
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(x, y)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is not None
        assert loss.item() > 0

    def test_gpt2_backward(self):
        config = GPT2Config.gpt2_124m()
        model = GPT2Model.from_config(config)

        batch_size = 2
        seq_len = 16
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(x, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gpt2_config_variants(self):
        configs = [
            GPT2Config.gpt2_124m(),
            GPT2Config.gpt2_355m(),
            GPT2Config.gpt2_774m(),
            GPT2Config.gpt2_1_5b(),
        ]

        for cfg in configs:
            model = GPT2Model.from_config(cfg)
            x = torch.randint(0, cfg.vocab_size, (1, 32))
            logits, _ = model(x)
            assert logits.shape == (1, 32, cfg.vocab_size)

    def test_model_device_transfer(self):
        config = GPT2Config.gpt2_124m()
        model = GPT2Model.from_config(config)

        x = torch.randint(0, config.vocab_size, (1, 16))

        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            logits, _ = model(x)
            assert logits.is_cuda

        if torch.backends.mps.is_available():
            model = model.to("mps")
            x = x.to("mps")
            logits, _ = model(x)
            assert logits.device.type == "mps"
