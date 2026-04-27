# ollama-classifier

A Python wrapper around the Ollama Python SDK for text classification with constrained output and confidence scoring. **Now supports multiple inference backends**: Ollama, vLLM, SGLang, and llama.cpp.

## Features

- **Constrained Output**: Uses JSON schema with enum constraints to ensure only valid choices are generated
- **Confidence Scoring**: Multi-call evaluation with softmax for calibrated probabilities
- **Sync & Async**: Full support for both synchronous and asynchronous operations
- **Batch Processing**: Classify multiple texts efficiently
- **Flexible Choices**: Support for simple labels or labels with descriptions
- **Custom Prompts**: Override the default system prompt for specialized tasks
- **Multiple Backends**: Use Ollama, vLLM, SGLang, or llama.cpp as your inference engine (local or remote)

## Installation

### Core (Ollama only)

```bash
pip install ollama-classifier
```

Or with uv:

```bash
uv add ollama-classifier
```

### With additional backends (vLLM, SGLang, llama.cpp)

```bash
pip install "ollama-classifier[backends]"
```

Or with uv:

```bash
uv add "ollama-classifier[backends]"
```

## Prerequisites

- **Ollama backend**: [Ollama](https://ollama.com/download) installed and running, with a model pulled (e.g., `ollama pull llama3.2`)
- **vLLM backend**: A running [vLLM](https://docs.vllm.ai/) server
- **SGLang backend**: A running [SGLang](https://sglang.ai/) server
- **llama.cpp backend**: A running [llama.cpp server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)

## Quick Start

### Ollama (original backend)

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

### vLLM

```python
from ollama_classifier.backends import VLLMBackend
from ollama_classifier import LLMClassifier

backend = VLLMBackend(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:8000/v1",
)
classifier = LLMClassifier(backend)

result = classifier.classify(
    text="I love this product!",
    choices=["positive", "negative", "neutral"]
)
```

### SGLang

```python
from ollama_classifier.backends import SGLangBackend
from ollama_classifier import LLMClassifier

backend = SGLangBackend(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:30000/v1",
)
classifier = LLMClassifier(backend)

result = classifier.classify(
    text="I love this product!",
    choices=["positive", "negative", "neutral"]
)
```

### llama.cpp

```python
from ollama_classifier.backends import LlamaCppBackend
from ollama_classifier import LLMClassifier

backend = LlamaCppBackend(
    model="model",
    base_url="http://localhost:8080/v1",
)
classifier = LLMClassifier(backend)

result = classifier.classify(
    text="I love this product!",
    choices=["positive", "negative", "neutral"]
)
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

### Scoring (Multi-Call with Softmax)

Get calibrated probability distribution over all choices. Makes N API calls for N choices:

```python
result = classifier.score(
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

results = classifier.batch_classify(
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

## Inference Backends

### Ollama (default)

The original backend using the Ollama Python SDK. Requires Ollama installed locally.

```python
from ollama import Client
from ollama_classifier import OllamaClassifier

classifier = OllamaClassifier(Client(), model="llama3.2")
```

### vLLM

High-throughput serving engine. Supports local and remote servers.

**Local server:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 --port 8000
```

**Connect:**
```python
from ollama_classifier.backends import VLLMBackend
from ollama_classifier import LLMClassifier

backend = VLLMBackend(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:8000/v1",
)
classifier = LLMClassifier(backend)
```

**Remote server:**
```python
backend = VLLMBackend(
    model="your-model",
    base_url="https://your-vllm-server.com/v1",
    api_key="your-api-key",  # if authentication is required
)
```

### SGLang

Fast serving system for large language models. Supports local and remote servers.

**Local server:**
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 --port 30000
```

**Connect:**
```python
from ollama_classifier.backends import SGLangBackend
from ollama_classifier import LLMClassifier

backend = SGLangBackend(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:30000/v1",
)
classifier = LLMClassifier(backend)
```

### llama.cpp

Lightweight inference via `llama-server`. Ideal for CPU or mixed CPU/GPU environments.

**Local server:**
```bash
./llama-server -m model.gguf --host 0.0.0.0 --port 8080 -c 4096
```

**Connect:**
```python
from ollama_classifier.backends import LlamaCppBackend
from ollama_classifier import LLMClassifier

backend = LlamaCppBackend(
    model="model",
    base_url="http://localhost:8080/v1",
)
classifier = LLMClassifier(backend)
```

> **Note:** JSON schema constraints and logprobs require llama.cpp to be
> compiled with the appropriate flags (e.g., ``LLAMA_JSON_SCHEMA`` and
> ``LLAMA_SUPPORT_LOGPROBS``).

### Backend Configuration

All backends share common configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | *(required)* | Model identifier |
| `base_url` | Engine-specific | Base URL of the inference server |
| `api_key` | `"not-needed"` | API key for authentication |
| `timeout` | `120.0` | Request timeout in seconds |
| `max_tokens` | `256` | Maximum tokens to generate |
| `extra_body` | `{}` | Extra parameters merged into every request |

## API Reference

### ClassificationResult

```python
@dataclass
class ClassificationResult:
    prediction: str              # The predicted choice label
    confidence: float            # Confidence score (0.0 to 1.0)
    probabilities: Dict[str, float]  # Probability distribution over all choices
    raw_response: Dict           # Raw response for debugging
```

### OllamaClassifier Methods

| Method | Async | Description |
|--------|-------|-------------|
| `generate(text, choices, system_prompt)` | `agenerate` | Constrained output only (fastest) |
| `score(text, choices, system_prompt)` | `ascore` | Multi-call evaluation with softmax |
| `classify(text, choices, system_prompt)` | `aclassify` | Full classification with confidence scores |
| `batch_generate(texts, choices, system_prompt)` | `abatch_generate` | Batch constrained output |
| `batch_score(texts, choices, system_prompt)` | `abatch_score` | Batch scoring |
| `batch_classify(texts, choices, system_prompt)` | `abatch_classify` | Batch classification |

### LLMClassifier Methods

`LLMClassifier` exposes the **same** API as `OllamaClassifier` but accepts any `LLMBackend`:

| Method | Async | Description |
|--------|-------|-------------|
| `generate(text, choices, system_prompt)` | `agenerate` | Constrained output only (fastest) |
| `score(text, choices, system_prompt)` | `ascore` | Multi-call evaluation with softmax |
| `classify(text, choices, system_prompt)` | `aclassify` | Full classification with confidence scores |
| `batch_generate(texts, choices, system_prompt)` | `abatch_generate` | Batch constrained output |
| `batch_score(texts, choices, system_prompt)` | `abatch_score` | Batch scoring |
| `batch_classify(texts, choices, system_prompt)` | `abatch_classify` | Batch classification |

### Parameters

- **text** (str): The text to classify
- **texts** (List[str]): List of texts to classify (batch methods)
- **choices** (Union[List[str], Dict[str, str]]): Either a list of choice labels, or a dict mapping labels to descriptions
- **system_prompt** (str | None): Optional custom system prompt

## Choosing a Method

| Use Case | Recommended Method |
|----------|-------------------|
| Speed is critical, no confidence needed | `generate` |
| Accurate confidence scores | `classify` / `score` |
| Batch processing | `batch_classify` or `batch_score` |
| Concurrent processing | Async variants (`aclassify`, etc.) |

## License

MIT License

## Development

This project just started! Looking forward to suggestions, issues, and pull requests!
