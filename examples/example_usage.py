"""Example usage of the Ollama Classifier.

This script demonstrates various features of the ollama-classifier package:
- Basic classification
- Classification with label descriptions
- Fast vs complete scoring methods
- Batch classification
- Async usage
"""

import asyncio
from ollama import Client, AsyncClient
from ollama_classifier import OllamaClassifier, ClassificationResult


def basic_classification():
    """Basic text classification with simple choices."""
    print("\n" + "=" * 60)
    print("Basic Classification (Simple Choices)")
    print("=" * 60)
    
    client = Client()
    classifier = OllamaClassifier(client, model="llama3.2")
    
    text = "The new quantum processor architecture drastically reduces latency."
    
    result = classifier.classify(
        text=text,
        choices=["technology", "sports", "politics", "entertainment"],
    )
    
    print(f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Probabilities: {result.probabilities}")


def classification_with_descriptions():
    """Classification with label descriptions for better accuracy."""
    print("\n" + "=" * 60)
    print("Classification with Label Descriptions")
    print("=" * 60)
    
    client = Client()
    classifier = OllamaClassifier(client, model="llama3.2")
    
    text = "This restaurant has amazing food but terrible service."
    
    # Choices with descriptions help the model understand each category
    choices = {
        "positive": "Text expresses happiness, satisfaction, or approval",
        "negative": "Text expresses anger, disappointment, or disapproval",
        "mixed": "Text contains both positive and negative sentiments",
        "neutral": "Text is factual without strong emotional content",
    }
    
    result = classifier.classify(
        text=text,
        choices=choices,
    )
    
    print(f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Probabilities: {result.probabilities}")


def custom_system_prompt():
    """Classification with a custom system prompt."""
    print("\n" + "=" * 60)
    print("Classification with Custom System Prompt")
    print("=" * 60)
    
    client = Client()
    classifier = OllamaClassifier(client, model="llama3.2")
    
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


def fast_vs_complete_scoring():
    """Compare fast (single-call) vs complete (multi-call) scoring."""
    print("\n" + "=" * 60)
    print("Fast vs Complete Scoring")
    print("=" * 60)
    
    client = Client()
    classifier = OllamaClassifier(client, model="llama3.2")
    
    text = "The movie was absolutely fantastic!"
    choices = ["positive", "negative", "neutral"]
    
    # Fast scoring - single API call, extracts logprobs from token distribution
    print("\n--- Fast Scoring (single API call) ---")
    result_fast = classifier.score_fast(text=text, choices=choices)
    print(f"Prediction: {result_fast.prediction}")
    print(f"Confidence: {result_fast.confidence:.2%}")
    print(f"Probabilities: {result_fast.probabilities}")
    
    # Complete scoring - multiple API calls, more accurate probability distribution
    print("\n--- Complete Scoring (multiple API calls) ---")
    result_complete = classifier.score_complete(text=text, choices=choices)
    print(f"Prediction: {result_complete.prediction}")
    print(f"Confidence: {result_complete.confidence:.2%}")
    print(f"Probabilities: {result_complete.probabilities}")


def generate_only():
    """Generate constrained output without confidence scores (fastest)."""
    print("\n" + "=" * 60)
    print("Generate Only (Fastest - No Confidence Scores)")
    print("=" * 60)
    
    client = Client()
    classifier = OllamaClassifier(client, model="llama3.2")
    
    texts = [
        "The team won the championship!",
        "Stock prices plummeted after the announcement.",
        "Scientists discovered a new species in the Amazon.",
    ]
    
    choices = ["sports", "finance", "science", "politics"]
    
    for text in texts:
        prediction = classifier.generate(text=text, choices=choices)
        print(f"Text: {text}")
        print(f"Prediction: {prediction}\n")


def batch_classification():
    """Batch classification of multiple texts."""
    print("\n" + "=" * 60)
    print("Batch Classification")
    print("=" * 60)
    
    client = Client()
    classifier = OllamaClassifier(client, model="llama3.2")
    
    texts = [
        "The goalkeeper made an incredible save!",
        "The central bank raised interest rates.",
        "The new smartphone features a revolutionary camera.",
    ]
    
    choices = ["sports", "finance", "technology"]
    
    # Batch classify with fast scoring
    print("\n--- Batch Classify (Fast) ---")
    results = classifier.batch_classify(texts=texts, choices=choices)
    
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"  Prediction: {result.prediction} ({result.confidence:.2%})")
    
    # Batch classify with complete scoring
    print("\n--- Batch Classify (Complete) ---")
    results_complete = classifier.batch_classify_complete(texts=texts, choices=choices)
    
    for text, result in zip(texts, results_complete):
        print(f"Text: {text}")
        print(f"  Prediction: {result.prediction} ({result.confidence:.2%})")


async def async_classification():
    """Async classification example."""
    print("\n" + "=" * 60)
    print("Async Classification")
    print("=" * 60)
    
    client = AsyncClient()
    classifier = OllamaClassifier(client, model="llama3.2")
    
    # Single text async classification
    text = "The concert was an unforgettable experience!"
    choices = ["positive", "negative", "neutral"]
    
    result = await classifier.aclassify(text=text, choices=choices)
    
    print(f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Probabilities: {result.probabilities}")


async def async_batch_classification():
    """Async batch classification with concurrent execution."""
    print("\n" + "=" * 60)
    print("Async Batch Classification (Concurrent)")
    print("=" * 60)
    
    client = AsyncClient()
    classifier = OllamaClassifier(client, model="llama3.2")
    
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


def main():
    """Run all examples."""
    print("=" * 60)
    print("OLLAMA CLASSIFIER - EXAMPLE USAGE")
    print("=" * 60)
    
    # Sync examples
    basic_classification()
    classification_with_descriptions()
    custom_system_prompt()
    fast_vs_complete_scoring()
    generate_only()
    batch_classification()
    
    # Async examples
    print("\n" + "=" * 60)
    print("ASYNC EXAMPLES")
    print("=" * 60)
    
    asyncio.run(async_classification())
    asyncio.run(async_batch_classification())
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()