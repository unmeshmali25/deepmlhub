import argparse
import json
from pathlib import Path

from llm_training.data.messages import load_jsonl
from llm_training.evaluation.generation import generate_response
from llm_training.evaluation.model import load_model


def evaluate(
    model_name: str,
    eval_path: Path,
    output_path: Path,
    adapter_path: Path | None = None,
    limit: int | None = None,
) -> None:
    """Generate and save predictions for an evaluation dataset."""
    model, tokenizer = load_model(
        model_name=model_name,
        adapter_path=adapter_path,
    )

    records = load_jsonl(eval_path)

    if limit is not None:
        records = records[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, record in enumerate(records, start=1):
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=record["prompt"],
            )

            result = {
                "prompt": record["prompt"],
                "ground_truth": record["ground_truth"],
                "response": response,
            }

            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(
                f"Evaluated {index}/{len(records)}",
                flush=True,
            )


def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit", type=int)

    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        eval_path=args.eval_data,
        output_path=args.output,
        adapter_path=args.adapter,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
