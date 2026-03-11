# ollama-classifier

A Python wrapper around the Ollama Python SDK for text classification with constrained output and confidence scoring.

## Features

- **Constrained Output**: Uses JSON schema with enum constraints to ensure only valid choices are generated
- **Confidence Scoring**: Two methods available:
  - **Fast**: Single API call with logprob extraction
  - **Complete**: Multi-call evaluation with softmax for calibrated probabilities
- **Sync & Async**: Full support for both synchronous and asynchronous operations
- **Batch Processing**: Classify multiple texts efficiently
- **Flexible Choices**: Support for simple labels or labels with descriptions
- **Custom Prompts**: Override the default system prompt for specialized tasks

## Installation

```bash
pip install ollama-classifier
```

Or with uv:

```bash
uv add ollama-classifier
```

## Prerequisites

- [Ollama](https://ollama.com/download) installed and running
- A model pulled (e.g., `ollama pull llama3.2`)

## Quick Start

```python
from ollama import Client
from ollama_classifier import OllamaClassifier

client = Client()
classifier = OllamaClassifier(client, model="llama3.2")

result = classifier.classify(
    text="I love this product!",
    choices=["positive", "negative", "neutral"]
)

print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Probabilities: {result.probabilities}")
```

## Usage

### Basic Classification

```python
from ollama import Client
from ollama_classifier import OllamaClassifier

client = Client()
classifier = OllamaClassifier(client, model="llama3.2")

result = classifier.classify(
    text="The goalkeeper made an incredible save!",
    choices=["sports", "politics", "technology", "entertainment"]
)
```

### Classification with Label Descriptions

Providing descriptions helps the model understand each category better:

```python
choices = {
    "positive": "Text expresses happiness, satisfaction, or approval",
    "negative": "Text expresses anger, disappointment, or disapproval",
    "mixed": "Text contains both positive and negative sentiments",
    "neutral": "Text is factual without strong emotional content",
}

result = classifier.classify(
    text="The food was amazing but the service was terrible.",
    choices=choices
)
```

### Custom System Prompt

```python
result = classifier.classify(
    text="The quarterly earnings exceeded analyst expectations.",
    choices=["bullish", "bearish", "neutral"],
    system_prompt="You are a financial sentiment analyzer. "
                  "Classify financial news based on market sentiment."
)
```

### Scoring Methods

#### Fast Scoring (Single API Call)

```python
result = classifier.score_fast(
    text="The movie was fantastic!",
    choices=["positive", "negative", "neutral"]
)
```

#### Complete Scoring (Multi-Call with Softmax)

More accurate but makes N API calls for N choices:

```python
result = classifier.score_complete(
    text="The movie was fantastic!",
    choices=["positive", "negative", "neutral"]
)
```

### Generate Only (Fastest)

When you only need the prediction without confidence scores:

```python
prediction = classifier.generate(
    text="The team won the championship!",
    choices=["sports", "finance", "politics"]
)
```

### Batch Classification

```python
texts = [
    "The goalkeeper made an incredible save!",
    "The central bank raised interest rates.",
    "The new smartphone features a revolutionary camera.",
]

# Fast batch classification
results = classifier.batch_classify(
    texts=texts,
    choices=["sports", "finance", "technology"]
)

# Complete batch classification (more accurate)
results = classifier.batch_classify_complete(
    texts=texts,
    choices=["sports", "finance", "technology"]
)

for text, result in zip(texts, results):
    print(f"{text} -> {result.prediction} ({result.confidence:.2%})")
```

### Async Usage

```python
import asyncio
from ollama import AsyncClient
from ollama_classifier import OllamaClassifier

async def main():
    client = AsyncClient()
    classifier = OllamaClassifier(client, model="llama3.2")
    
    # Single classification
    result = await classifier.aclassify(
        text="The concert was amazing!",
        choices=["positive", "negative", "neutral"]
    )
    
    # Batch classification (concurrent)
    results = await classifier.abatch_classify(
        texts=["Text 1", "Text 2", "Text 3"],
        choices=["positive", "negative", "neutral"]
    )

asyncio.run(main())
```

## API Reference

### ClassificationResult

```python
@dataclass
class ClassificationResult:
    prediction: str              # The predicted choice label
    confidence: float            # Confidence score (0.0 to 1.0)
    probabilities: Dict[str, float]  # Probability distribution over all choices
    raw_response: Dict           # Raw Ollama response for debugging
```

### OllamaClassifier Methods

| Method | Async | Description |
|--------|-------|-------------|
| `generate(text, choices, system_prompt)` | `agenerate` | Constrained output only (fastest) |
| `score_fast(text, choices, system_prompt)` | `ascore_fast` | Single-call logprob extraction |
| `score_complete(text, choices, system_prompt)` | `ascore_complete` | Multi-call evaluation with softmax |
| `classify(text, choices, system_prompt)` | `aclassify` | Generate + score_fast |
| `classify_complete(text, choices, system_prompt)` | `aclassify_complete` | Generate + score_complete |
| `batch_generate(texts, choices, system_prompt)` | `abatch_generate` | Batch constrained output |
| `batch_score_fast(texts, choices, system_prompt)` | `abatch_score_fast` | Batch fast scoring |
| `batch_score_complete(texts, choices, system_prompt)` | `abatch_score_complete` | Batch complete scoring |
| `batch_classify(texts, choices, system_prompt)` | `abatch_classify` | Batch classify (fast) |
| `batch_classify_complete(texts, choices, system_prompt)` | `abatch_classify_complete` | Batch classify (complete) |

### Parameters

- **text** (str): The text to classify
- **texts** (List[str]): List of texts to classify (batch methods)
- **choices** (Union[List[str], Dict[str, str]]): Either a list of choice labels, or a dict mapping labels to descriptions
- **system_prompt** (str | None): Optional custom system prompt

## Choosing a Method

| Use Case | Recommended Method |
|----------|-------------------|
| Speed is critical, no confidence needed | `generate` |
| Speed with confidence scores | `classify` / `score_fast` |
| Accurate confidence scores | `classify_complete` / `score_complete` |
| Batch processing | `batch_classify` or `batch_classify_complete` |
| Concurrent processing | Async variants (`aclassify`, etc.) |

## License

MIT License

## Development

This project just started! Looking forward to suggestions, issues, and pull requests!