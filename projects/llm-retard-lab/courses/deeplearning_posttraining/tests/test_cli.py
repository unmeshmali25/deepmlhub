from llm_training.__main__ import TrainConfig, parse_args
import pytest
from pathlib import Path


def test_minimal_argv_applies_default():
    cfg = parse_args(["--data", "data.jsonl"])

    assert cfg.model_name == "Qwen/Qwen2.5-1.5B"
    assert cfg.data_path == Path("data.jsonl")
    assert cfg.output_dir == Path("./adapter_out")
    assert cfg.lora_r == 16
    assert cfg.learning_rate == 2e-4
    assert cfg.num_train_epochs == 3
    assert cfg.per_device_train_batch_size == 4
    assert cfg.bf16 is True
    assert cfg.fp16 is False
    assert cfg.max_steps is None
    assert cfg.seed == 42


def test_cli_names_map_to_config_fields():
    cfg = parse_args(
        [
            "--data",
            "train.jsonl",
            "--lora-r",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--grad-accum",
            "2",
            "--lr",
            "1e-4",
            "--max-length",
            "512",
        ]
    )

    assert cfg.lora_r == 8
    assert cfg.num_train_epochs == 1
    assert cfg.per_device_train_batch_size == 2
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.learning_rate == 1e-4
    assert cfg.max_length == 512


def test_data_is_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_argparse_coerces_types():
    cfg = parse_args(
        [
            "--data",
            "data.jsonl",
            "--lora-r",
            "8",
            "--lora-dropout",
            "0.1",
            "--lr",
            "3e-4",
        ]
    )

    assert cfg.lora_r == 8
    assert isinstance(cfg.lora_r, int)
    assert cfg.lora_dropout == pytest.approx(0.1)
    assert cfg.learning_rate == pytest.approx(3e-4)


def test_path_flags_coerce_to_path():
    cfg = parse_args(["--data", "data.jsonl", "--output", "out/"])

    assert cfg.data_path == Path("data.jsonl")
    assert cfg.output_dir == Path("out/")


def test_bf16_defaults_true():
    cfg = parse_args(["--data", "data.jsonl"])
    assert cfg.bf16 is True


def test_no_bf16_disables_bf16():
    cfg = parse_args(["--data", "data.jsonl", "--no-bf16"])
    assert cfg.bf16 is False


def test_fp16_defaults_false_and_enables():
    cfg = parse_args(["--data", "data.jsonl"])
    assert cfg.fp16 is False

    cfg = parse_args(["--data", "data.jsonl", "--fp16"])
    assert cfg.fp16 is True


def test_max_steps_defaults_none():
    cfg = parse_args(["--data", "data.jsonl"])
    assert cfg.max_steps is None


def test_max_steps_overrides():
    cfg = parse_args(["--data", "data.jsonl", "--max-steps", "2"])
    assert cfg.max_steps == 2


def test_unknown_flag_exits():
    with pytest.raises(SystemExit):
        parse_args(["--data", "data.jsonl", "--bogus"])
