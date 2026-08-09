from llm_training.config import TrainConfig
from llm_training.data.messages import to_sft_dataset


def train(cfg: TrainConfig) -> None:
    # Heavy imports deferred to call-time: peft/trl/transformers only exist
    # in the training venv (pod), NOT the test venv. Importing them here keeps
    # this module cheap to import, so the CLI tests can still parse args.
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # data
    train_ds, eval_ds = to_sft_dataset(cfg.data_path)
    train_ds = train_ds.remove_columns(["prompt", "response"])
    eval_ds = eval_ds.remove_columns(["prompt", "response"])

    # LoRA bypasses
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training args
    training_args = SFTConfig(
        output_dir=str(cfg.output_dir),
        max_length=cfg.max_length,
        packing=False,
        completion_only_loss=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        seed=cfg.seed,
        report_to=cfg.report_to,
    )

    # trainer
    trainer = SFTTrainer(
        model=cfg.model_name,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
