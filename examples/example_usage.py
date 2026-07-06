"""Example usage of ollama-classifier with multiple backends.

This script demonstrates the v0.4.0 unified architecture:

- All backends are used the same way: create a backend, wrap it in
  ``LLMClassifier``, and call ``generate()`` or ``classify()``.
- Two scoring methods:

  * ``generate()`` — adaptive constrained generation (1 to ``max_calls``
    API calls). Fast, approximate.
  * ``classify()`` — exact multi-call completion scoring (N calls for N
    labels). Gold-standard confidence.

Run with::

    ollama pull llama3.2
    python -m examples.example_usage
"""

import asyncio

from ollama_classifier import LLMClassifier
from ollama_classifier.backends import OllamaBackend


def make_classifier() -> LLMClassifier:
    """Build an ``LLMClassifier`` backed by Ollama."""
    backend = OllamaBackend(model="llama3.2")
    return LLMClassifier(backend)


def basic_classification() -> None:
    """Basic text classification with simple choices (exact multi-call)."""
    print("\n" + "=" * 60)
    print("Basic Classification (classify — multi_call)")
    print("=" * 60)

    classifier = make_classifier()
    text = "The new quantum processor architecture drastically reduces latency."

    result = classifier.classify(
        text=text,
        choices=["technology", "sports", "politics", "entertainment"],
    )

    print(f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Method: {result.method}")
    print(f"Probabilities: {result.probabilities}")


def classification_with_descriptions() -> None:
    """Classification with label descriptions for better accuracy."""
    print("\n" + "=" * 60)
    print("Classification with Label Descriptions")
    print("=" * 60)

    classifier = make_classifier()
    text = "This restaurant has amazing food but terrible service."

    # Choices with descriptions help the model understand each category
    choices = {
        "positive": "Text expresses happiness, satisfaction, or approval",
        "negative": "Text expresses anger, disappointment, or disapproval",
        "mixed": "Text contains both positive and negative sentiments",
        "neutral": "Text is factual without strong emotional content",
    }

    result = classifier.classify(text=text, choices=choices)

    print(f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Probabilities: {result.probabilities}")


def custom_system_prompt() -> None:
    """Classification with a custom system prompt."""
    print("\n" + "=" * 60)
    print("Classification with Custom System Prompt")
    print("=" * 60)

    classifier = make_classifier()
    text = "The quarterly earnings exceeded analyst expectations."

    result = classifier.classify(
        text=text,
        choices=["bullish", "bearish", "neutral"],
        system_prompt="You are a financial sentiment analyzer. "
                      "Classify financial news based on market sentiment.",
    )

    print(f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Probabilities: {result.probabilities}")


def adaptive_generate() -> None:
    """Adaptive constrained generation with budget-controlled max_calls."""
    print("\n" + "=" * 60)
    print("Adaptive Generation (generate — max_calls)")
    print("=" * 60)

    classifier = make_classifier()

    texts = [
        "The team won the championship!",
        "Stock prices plummeted after the announcement.",
        "Scientists discovered a new species in the Amazon.",
    ]
    choices = ["sports", "finance", "science", "politics"]

    for text in texts:
        # max_calls=1 -> single constrained call, fast, approximate
        result = classifier.generate(text=text, choices=choices, max_calls=1)
        print(f"Text: {text}")
        print(f"  Prediction: {result.prediction}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Approximate: {result.approximate}")
        print(f"  Coverage: {result.coverage}")
        print(f"  Calls: {result.n_calls}\n")


def exact_generate() -> None:
    """Fully resolved generation (max_calls=None = exact)."""
    print("\n" + "=" * 60)
    print("Exact Generation (generate — max_calls=None)")
    print("=" * 60)

    classifier = make_classifier()
    text = "The team won the championship!"
    choices = ["sports", "finance", "science", "politics"]

    result = classifier.generate(text=text, choices=choices, max_calls=None)
    print(f"Text: {text}")
    print(f"  Prediction: {result.prediction}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Approximate: {result.approximate}")
    print(f"  Calls: {result.n_calls}")


def batch_classification() -> None:
    """Batch classification of multiple texts (parallelized)."""
    print("\n" + "=" * 60)
    print("Batch Classification (batch_classify)")
    print("=" * 60)

    classifier = make_classifier()

    texts = [
        "The goalkeeper made an incredible save!",
        "The central bank raised interest rates.",
        "The new smartphone features a revolutionary camera.",
    ]
    choices = ["sports", "finance", "technology"]

    # Batch classify with calibrated confidence scores (parallelized)
    results = classifier.batch_classify(texts=texts, choices=choices)

    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"  Prediction: {result.prediction} ({result.confidence:.2%})")


def batch_generate() -> None:
    """Batch adaptive generation (parallelized)."""
    print("\n" + "=" * 60)
    print("Batch Generation (batch_generate, max_calls=1)")
    print("=" * 60)

    classifier = make_classifier()

    texts = [
        "The goalkeeper made an incredible save!",
        "The central bank raised interest rates.",
        "The new smartphone features a revolutionary camera.",
    ]
    choices = ["sports", "finance", "technology"]

    results = classifier.batch_generate(
        texts=texts, choices=choices, max_calls=1
    )

    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"  Prediction: {result.prediction} ({result.confidence:.2%})")


async def async_classification() -> None:
    """Async classification example."""
    print("\n" + "=" * 60)
    print("Async Classification")
    print("=" * 60)

    classifier = make_classifier()
    text = "The concert was an unforgettable experience!"
    choices = ["positive", "negative", "neutral"]

    result = await classifier.aclassify(text=text, choices=choices)

    print(f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Probabilities: {result.probabilities}")


async def async_batch_classification() -> None:
    """Async batch classification with concurrent execution."""
    print("\n" + "=" * 60)
    print("Async Batch Classification (Concurrent)")
    print("=" * 60)

    classifier = make_classifier()

    texts = [
        "The team secured a decisive victory.",
        "Markets rallied on positive economic data.",
        "The software update fixes critical security vulnerabilities.",
    ]
    choices = ["sports", "finance", "technology"]

    # All texts are processed concurrently
    results = await classifier.abatch_classify(texts=texts, choices=choices)

    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"  Prediction: {result.prediction} ({result.confidence:.2%})")


def vllm_example() -> None:
    """Classification using vLLM backend."""
    print("\n" + "=" * 60)
    print("vLLM Backend Classification")
    print("=" * 60)

    from ollama_classifier.backends import VLLMBackend

    backend = VLLMBackend(
        model="meta-llama/Llama-3.2-3B-Instruct",
        base_url="http://localhost:8000/v1",
    )
    classifier = LLMClassifier(backend)

    text = "The new quantum processor architecture drastically reduces latency."
    result = classifier.classify(
        text=text,
        choices=["technology", "sports", "politics", "entertainment"],
    )

    print(f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Probabilities: {result.probabilities}")


def sglang_example() -> None:
    """Classification using SGLang backend."""
    print("\n" + "=" * 60)
    print("SGLang Backend Classification")
    print("=" * 60)

    from ollama_classifier.backends import SGLangBackend

    backend = SGLangBackend(
        model="meta-llama/Llama-3.2-3B-Instruct",
        base_url="http://localhost:30000/v1",
    )
    classifier = LLMClassifier(backend)

    text = "The central bank raised interest rates by 50 basis points."
    result = classifier.classify(
        text=text,
        choices=["sports", "finance", "technology", "entertainment"],
    )

    print(f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Probabilities: {result.probabilities}")


def llamacpp_example() -> None:
    """Classification using llama.cpp backend."""
    print("\n" + "=" * 60)
    print("llama.cpp Backend Classification")
    print("=" * 60)

    from ollama_classifier.backends import LlamaCppBackend

    backend = LlamaCppBackend(
        model="model",
        base_url="http://localhost:8080/v1",
    )
    classifier = LLMClassifier(backend)

    text = "The goalkeeper made an incredible save!"
    result = classifier.classify(
        text=text,
        choices=["sports", "finance", "technology", "entertainment"],
    )

    print(f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Probabilities: {result.probabilities}")


def main() -> None:
    """Run all examples."""
    print("=" * 60)
    print("OLLAMA CLASSIFIER — EXAMPLE USAGE (v0.4.0)")
    print("=" * 60)

    # Sync examples (Ollama backend)
    basic_classification()
    classification_with_descriptions()
    custom_system_prompt()
    adaptive_generate()
    exact_generate()
    batch_classification()
    batch_generate()

    # Async examples (Ollama backend)
    print("\n" + "=" * 60)
    print("ASYNC EXAMPLES")
    print("=" * 60)

    asyncio.run(async_classification())
    asyncio.run(async_batch_classification())

    # Backend examples (require running servers)
    print("\n" + "=" * 60)
    print("BACKEND EXAMPLES (vLLM, SGLang, llama.cpp)")
    print("=" * 60)
    print("Note: These require running inference servers.")
    print("Uncomment the relevant function calls below to try them.\n")

    # vllm_example()       # Requires: vllm server on localhost:8000
    # sglang_example()     # Requires: sglang server on localhost:30000
    # llamacpp_example()   # Requires: llama-server on localhost:8080

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
