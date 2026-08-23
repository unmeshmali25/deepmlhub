import re
from typing import Any


def normalize(text: str) -> str:
    """Normalize text for simple fact matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_record(record: dict[str, Any]) -> dict[str, bool]:
    """Score policy facts in one generated response."""
    response = normalize(record["response"])
    truth = record["ground_truth"]

    has_refund = "refund" in response

    if truth["in_window"]:
        correct_window = ("within" in response or "eligible" in response) and (
            "30-day" in response or "30 day" in response
        )
    else:
        correct_window = (
            "outside" in response
            or "not eligible" in response
            or "return window has passed" in response
        )

    if truth["expect_refund"]:
        correct_refund = has_refund
    else:
        correct_refund = (
            "not eligible" in response
            or "cannot refund" in response
            or "no refund" in response
        )

    expected_method = truth.get("refund_method")

    if expected_method is None:
        correct_method = None
    else:
        expected_method = expected_method.lower()

        if expected_method == "original payment method":
            correct_method = (
                "original payment" in response or "original method" in response
            )
        else:
            correct_method = expected_method in response

    expected_days = str(truth["days_since_purchase"])
    correct_days = expected_days in response

    return {
        "correct_window": correct_window,
        "correct_refund": correct_refund,
        "correct_method": correct_method,
        "correct_days": correct_days,
    }


def summarize(
    scores: list[dict[str, bool | None]],
) -> dict[str, float | None]:
    """Calculate accuracy while excluding not-applicable metrics."""
    if not scores:
        return {}

    summary: dict[str, float | None] = {}

    for metric in scores[0]:
        values = [score[metric] for score in scores if score[metric] is not None]

        summary[metric] = sum(values) / len(values) if values else None

    return summary
