from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path


from llm_training.config import TrainConfig
from llm_training.training.sft_cot_lora import train


log = logging.getLogger("llm_training")


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """ """
    p = argparse.ArgumentParser(
        prog="llm_training", description="LoRA SFT - cold-start CoT on Qwen2.5"
    )

    # -- I/O --
    p.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B", help="HF model id or local_path"
    )
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Training .jsonl (one {user,cot,reply} per line)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("./adapter_out"),
        help="Where to save the LoRA adapter",
    )

    # --- LoRA ---
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # --- SFT ---
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--batch-size", type=int, default=4, help="per-device train batch size"
    )
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override epochs (integration tests use ~2)",
    )
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)

    # --- dtype flags (mutually-agnostic booleans) ---
    # `--bf16` is on by default; `--no-bf16` flips it off without --bf16.
    p.add_argument("--bf16", dest="bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--fp16", action="store_true", default=False)

    # --- misc + logging ---
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-to", default="none", help="wandb|mlflow|none")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    args = p.parse_args(argv)

    return TrainConfig(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        report_to=args.report_to,
    )


def main(argv: list[str] | None = None) -> int:
    """
    Return an exit code, 0=ok, 1=generic, 2=missing file
    """
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(
        level=logging.DEBUG if {"-v", "--verbose"} & set(argv) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = parse_args(argv)
    log.info("config=%s", cfg)

    try:
        train(cfg)
    except FileNotFoundError:
        log.error("training data not found: %s", cfg.data_path)
        return 2  # dedicated exit code = "missing input"
    except Exception:
        log.exception("training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
