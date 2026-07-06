"""Example: using the sample datasets from examples.sample_data.

Run this script with Ollama installed and a model pulled::

    ollama pull llama3.2
    python -m examples.run_sample_data
"""

from ollama_classifier import LLMClassifier
from ollama_classifier.backends import OllamaBackend

from examples.sample_data import (
    DATASET_WITHOUT_DESCRIPTIONS,
    DATASET_WITH_DESCRIPTIONS,
)


def evaluate(results, expected_labels):
    """Print results and compute accuracy."""
    # Each result is a ClassificationResult; extract .prediction for comparison
    correct = sum(r.prediction == e for r, e in zip(results, expected_labels))
    total = len(expected_labels)
    print(f"Accuracy: {correct}/{total} ({correct / total:.0%})\n")


def main():
    backend = OllamaBackend(model="llama3.2")
    classifier = LLMClassifier(backend)

    # ── Dataset without descriptions ────────────────────────────────────
    print("=" * 60)
    print("Dataset WITHOUT descriptions (generate, max_calls=1)")
    print("=" * 60)
    print(DATASET_WITHOUT_DESCRIPTIONS.description)
    print()

    # Adaptive generation: single call per text, fast & approximate
    results = classifier.batch_generate(
        texts=DATASET_WITHOUT_DESCRIPTIONS.texts,
        choices=DATASET_WITHOUT_DESCRIPTIONS.choices,
    )
    for text, result, expected in zip(
        DATASET_WITHOUT_DESCRIPTIONS.texts,
        results,
        DATASET_WITHOUT_DESCRIPTIONS.expected_labels,
    ):
        mark = "✓" if result.prediction == expected else "✗"
        print(f"  {mark} {text}")
        print(f"    Predicted: {result.prediction}  |  Expected: {expected}\n")

    evaluate(results, DATASET_WITHOUT_DESCRIPTIONS.expected_labels)

    # ── Dataset with descriptions (exact multi-call) ─────────────────────
    print("=" * 60)
    print("Dataset WITH descriptions (classify — multi_call)")
    print("=" * 60)
    print(DATASET_WITH_DESCRIPTIONS.description)
    print()

    # classify(): exact multi-call completion scoring (gold-standard)
    results_desc = classifier.batch_classify(
        texts=DATASET_WITH_DESCRIPTIONS.texts,
        choices=DATASET_WITH_DESCRIPTIONS.choices,
    )
    for text, result, expected in zip(
        DATASET_WITH_DESCRIPTIONS.texts,
        results_desc,
        DATASET_WITH_DESCRIPTIONS.expected_labels,
    ):
        mark = "✓" if result.prediction == expected else "✗"
        print(f"  {mark} {text}")
        print(f"    Predicted: {result.prediction}  |  Expected: {expected}\n")

    evaluate(results_desc, DATASET_WITH_DESCRIPTIONS.expected_labels)


if __name__ == "__main__":
    main()
