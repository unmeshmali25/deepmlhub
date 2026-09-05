import argparse
import json
from pathlib import Path

from llm_training.evaluation.metrics import score_record, summarize


def main() -> None:
    """Score saved evaluation predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
    ]

    scores = [score_record(record) for record in records]
    metrics = summarize(scores)

    args.output.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
