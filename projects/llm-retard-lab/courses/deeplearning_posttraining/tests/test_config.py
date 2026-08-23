from pathlib import Path

from llm_training.config import load_train_config


def test_load_train_config_reads_yaml_values():
    config_path = Path(__file__).parents[1] / "configs" / "config.yaml"

    cfg = load_train_config(config_path)

    assert cfg.model_name == "Qwen/Qwen2.5-1.5B"
    assert cfg.data_path == (
        Path(__file__).parents[1] / "data" / "01_cold_start_cot_sft" / "data.jsonl"
    )
    assert cfg.output_dir == (
        Path(__file__).parents[1] / "models" / "adapter_experiment"
    )

    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.learning_rate == 0.0002
    assert cfg.max_length == 1024

    assert cfg.warmup_ratio == 0.03
    assert cfg.lr_scheduler_type == "cosine"
    assert cfg.logging_steps == 10
    assert cfg.eval_steps == 50
    assert cfg.save_steps == 200
    assert cfg.save_total_limit == 2
    assert cfg.max_steps is None

    assert cfg.fp16 is False
    assert cfg.bf16 is True
    assert cfg.report_to == "none"
