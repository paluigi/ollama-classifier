"""Example: using the sample datasets from examples.sample_data.

Run this script with Ollama installed and a model pulled::

    ollama pull llama3.2
    python -m examples.run_sample_data
"""

from ollama import Client
from ollama_classifier import OllamaClassifier

from examples.sample_data import (
    DATASET_WITHOUT_DESCRIPTIONS,
    DATASET_WITH_DESCRIPTIONS,
)


def evaluate(results, expected_labels):
    """Print results and compute accuracy."""
    correct = sum(r == e for r, e in zip(results, expected_labels))
    total = len(expected_labels)
    print(f"Accuracy: {correct}/{total} ({correct / total:.0%})\n")


def main():
    client = Client()
    classifier = OllamaClassifier(client, model="llama3.2")

    # ── Dataset without descriptions ────────────────────────────────────
    print("=" * 60)
    print("Dataset WITHOUT descriptions")
    print("=" * 60)
    print(DATASET_WITHOUT_DESCRIPTIONS.description)
    print()

    predictions = classifier.batch_generate(
        texts=DATASET_WITHOUT_DESCRIPTIONS.texts,
        choices=DATASET_WITHOUT_DESCRIPTIONS.choices,
    )
    for text, pred, expected in zip(
        DATASET_WITHOUT_DESCRIPTIONS.texts,
        predictions,
        DATASET_WITHOUT_DESCRIPTIONS.expected_labels,
    ):
        mark = "✓" if pred == expected else "✗"
        print(f"  {mark} {text}")
        print(f"    Predicted: {pred}  |  Expected: {expected}\n")

    evaluate(predictions, DATASET_WITHOUT_DESCRIPTIONS.expected_labels)

    # ── Dataset with descriptions ────────────────────────────────────────
    print("=" * 60)
    print("Dataset WITH descriptions")
    print("=" * 60)
    print(DATASET_WITH_DESCRIPTIONS.description)
    print()

    predictions_desc = classifier.batch_generate(
        texts=DATASET_WITH_DESCRIPTIONS.texts,
        choices=DATASET_WITH_DESCRIPTIONS.choices,
    )
    for text, pred, expected in zip(
        DATASET_WITH_DESCRIPTIONS.texts,
        predictions_desc,
        DATASET_WITH_DESCRIPTIONS.expected_labels,
    ):
        mark = "✓" if pred == expected else "✗"
        print(f"  {mark} {text}")
        print(f"    Predicted: {pred}  |  Expected: {expected}\n")

    evaluate(predictions_desc, DATASET_WITH_DESCRIPTIONS.expected_labels)


if __name__ == "__main__":
    main()
