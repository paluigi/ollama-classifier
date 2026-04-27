"""Example usage of the Ollama Classifier with multiple backends.

This script demonstrates various features of the ollama-classifier package:
- Basic classification with Ollama
- Classification with label descriptions
- Batch classification
- Async usage
- Usage with vLLM, SGLang, and llama.cpp backends
"""

import asyncio
from ollama import Client, AsyncClient
from ollama_classifier import OllamaClassifier, LLMClassifier, ClassificationResult


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


def scoring():
    """Score text to get probability distribution over choices."""
    print("\n" + "=" * 60)
    print("Scoring (Multi-call evaluation with softmax)")
    print("=" * 60)
    
    client = Client()
    classifier = OllamaClassifier(client, model="llama3.2")
    
    text = "The movie was absolutely fantastic!"
    choices = ["positive", "negative", "neutral"]
    
    # Score uses multi-call evaluation for calibrated probabilities
    result = classifier.score(text=text, choices=choices)
    print(f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Probabilities: {result.probabilities}")


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
    
    # Batch classify with calibrated confidence scores
    results = classifier.batch_classify(texts=texts, choices=choices)
    
    for text, result in zip(texts, results):
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


def vllm_example():
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


def sglang_example():
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


def llamacpp_example():
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


def main():
    """Run all examples."""
    print("=" * 60)
    print("OLLAMA CLASSIFIER - EXAMPLE USAGE")
    print("=" * 60)
    
    # Sync examples (Ollama backend)
    basic_classification()
    classification_with_descriptions()
    custom_system_prompt()
    scoring()
    generate_only()
    batch_classification()
    
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
