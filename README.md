# ollama-classifier

A Python library for LLM-based text classification with constrained output and confidence scoring. **Supports multiple inference backends**: Ollama (≥0.12), vLLM, SGLang, and llama.cpp.

Every backend works through a single unified `LLMClassifier` with two scoring methods — adaptive constrained generation (`generate()`) and exact multi-call completion scoring (`classify()`).

## Features

- **Two Scoring Methods**: `generate()` for adaptive budget-controlled scoring, `classify()` for exact gold-standard confidence
- **Constrained Output**: Output is guaranteed to be one of your labels (JSON enum, `structured_outputs.choice`, regex, or GBNF — depending on backend)
- **Calibrated Confidence**: Probability distribution over all choices with geometric-mean normalization (no token-count bias)
- **Sync & Async**: Full support for both synchronous and asynchronous operations
- **Batch Processing**: Classify multiple texts efficiently with parallel execution
- **Flexible Choices**: Support for simple labels or labels with descriptions
- **Custom Prompts**: Override the default system prompt for specialized tasks
- **Multiple Backends**: Ollama, vLLM, SGLang, or llama.cpp — all behind one API

## Installation

```bash
pip install ollama-classifier
```

Or with uv:

```bash
uv add ollama-classifier
```

### Ollama runtime support (optional)

To use the **Ollama** backend, install the `ollama` SDK as an extra:

```bash
pip install "ollama-classifier[ollama]"
```

> **Note:** `httpx` and `pydantic` are required dependencies and installed automatically. The `ollama` Python SDK is optional — it is only needed when using `OllamaBackend`.

## Prerequisites

You need at least one running inference backend:

- **Ollama backend**: [Ollama](https://ollama.com/download) ≥0.12 installed and running, with a model pulled (e.g., `ollama pull llama3.2`). Logprobs support requires v0.12 or later.
- **vLLM backend**: A running [vLLM](https://docs.vllm.ai/) server
- **SGLang backend**: A running [SGLang](https://sglang.ai/) server
- **llama.cpp backend**: A running [llama.cpp server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)

## Quick Start

All backends follow the same pattern: create a backend, wrap it in an `LLMClassifier`, and call `generate()` or `classify()`.

### Ollama

```python
from ollama_classifier import LLMClassifier
from ollama_classifier.backends import OllamaBackend

backend = OllamaBackend(model="llama3.2")
classifier = LLMClassifier(backend)

result = classifier.classify(
    text="I love this product!",
    choices=["positive", "negative", "neutral"],
)

print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Method: {result.method}")
```

### vLLM

```python
from ollama_classifier import LLMClassifier
from ollama_classifier.backends import VLLMBackend

backend = VLLMBackend(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:8000/v1",
)
classifier = LLMClassifier(backend)

result = classifier.classify(
    text="I love this product!",
    choices=["positive", "negative", "neutral"],
)
```

### SGLang

```python
from ollama_classifier import LLMClassifier
from ollama_classifier.backends import SGLangBackend

backend = SGLangBackend(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:30000/v1",
)
classifier = LLMClassifier(backend)
```

### llama.cpp

```python
from ollama_classifier import LLMClassifier
from ollama_classifier.backends import LlamaCppBackend

backend = LlamaCppBackend(
    model="model",
    base_url="http://localhost:8080/v1",
)
classifier = LLMClassifier(backend)
```

## Choosing a Scoring Method

`LLMClassifier` provides two scoring methods. Each returns a `ClassificationResult` with a full probability distribution.

| | `generate()` | `classify()` |
|---|---|---|
| **How it works** | Adaptive constrained generation over a prefix trie of label tokens. Per-label logprobs are reconstructed from the winning generation path and any unresolved clusters. | Multi-call completion scoring: each label is scored independently as a completion of the prompt. |
| **API calls** | 1 to `max_calls` (adaptive) | N calls for N labels |
| **Confidence** | Divergence-aware (may be partial for labels that diverge early) | Exact (geometric-mean normalization) |
| **Approximate?** | `result.approximate` is `True` when any label has partial coverage | Always `False` — fully resolved |
| **`result.method`** | `"adaptive_generate"` | `"multi_call"` |
| **Best for** | Speed, large label sets, when a single call suffices | Gold-standard accuracy, small-to-medium label sets |

### `generate()` — adaptive, budget-controlled

The `max_calls` parameter controls the accuracy/cost tradeoff:

| `max_calls` | Behavior | Calls |
|---|---|---|
| `1` *(default)* | Single constrained call. Labels are scored up to their divergence point from the winning path. Fast but approximate — set `result.approximate=True`. | 1 |
| `K` | Adaptive resolution. After each call, unresolved label clusters trigger supplementary constrained calls until the budget is exhausted. | ≤ K |
| `None` | Fully recursive resolution. Every cluster is resolved to completion. Equivalent to exact scoring. | ≤ N |

```python
# Fast: single call, approximate confidence
result = classifier.generate(
    text="The team won the championship!",
    choices=["sports", "finance", "politics"],
    max_calls=1,
)
print(result.prediction)        # "sports"
print(result.approximate)       # True (if any label had partial coverage)
print(result.coverage)          # {"sports": 1.0, "finance": 1.0, "politics": 1.0}

# Adaptive: allow up to 3 calls for better resolution
result = classifier.generate(text="...", choices=[...], max_calls=3)

# Exact: resolve everything (unlimited calls)
result = classifier.generate(text="...", choices=[...], max_calls=None)
```

### `classify()` — exact, N calls

Makes one completion-scoring call per label. Each label's per-token logprobs are extracted via echo/prefill (vLLM, SGLang) or forced constrained generation (Ollama, llama.cpp), then normalized via geometric mean to eliminate token-count bias. This is the gold-standard confidence method.

```python
result = classifier.classify(
    text="The movie was fantastic!",
    choices=["positive", "negative", "neutral"],
)
print(result.method)        # "multi_call"
print(result.approximate)   # False
print(result.n_calls)       # 3
```

## Usage

### Basic Classification

```python
from ollama_classifier import LLMClassifier
from ollama_classifier.backends import OllamaBackend

backend = OllamaBackend(model="llama3.2")
classifier = LLMClassifier(backend)

result = classifier.classify(
    text="The goalkeeper made an incredible save!",
    choices=["sports", "politics", "technology", "entertainment"],
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
    choices=choices,
)
```

### Custom System Prompt

```python
result = classifier.classify(
    text="The quarterly earnings exceeded analyst expectations.",
    choices=["bullish", "bearish", "neutral"],
    system_prompt="You are a financial sentiment analyzer. "
                  "Classify financial news based on market sentiment.",
)
```

### Adaptive Generation

```python
# Single call, approximate (fastest)
result = classifier.generate(
    text="The team won the championship!",
    choices=["sports", "finance", "politics"],
)

print(result.prediction)
print(result.confidence)
print(result.coverage)  # per-label fraction of tokens scored
```

### Batch Processing

```python
texts = [
    "The goalkeeper made an incredible save!",
    "The central bank raised interest rates.",
    "The new smartphone features a revolutionary camera.",
]

results = classifier.batch_classify(
    texts=texts,
    choices=["sports", "finance", "technology"],
)

for text, result in zip(texts, results):
    print(f"{text} -> {result.prediction} ({result.confidence:.2%})")
```

### Async Usage

```python
import asyncio
from ollama_classifier import LLMClassifier
from ollama_classifier.backends import OllamaBackend

async def main():
    backend = OllamaBackend(model="llama3.2")
    classifier = LLMClassifier(backend)

    # Single classification
    result = await classifier.aclassify(
        text="The concert was amazing!",
        choices=["positive", "negative", "neutral"],
    )

    # Batch classification (concurrent)
    results = await classifier.abatch_classify(
        texts=["Text 1", "Text 2", "Text 3"],
        choices=["positive", "negative", "neutral"],
    )

asyncio.run(main())
```

## Inference Backends

### Ollama

Uses the Ollama Python SDK. Constraint mechanism: **JSON Schema enum** via the `format` parameter (the model generates `{"label": "<chosen>"}`). Requires Ollama runtime ≥0.12 for logprobs. Since modern Ollama removed `/api/tokenize` and never supported insert on instruct models, both tokenization and completion scoring use empirical forced constrained generation (forcing a label as the only valid choice).

```python
from ollama_classifier import LLMClassifier
from ollama_classifier.backends import OllamaBackend

# Local
backend = OllamaBackend(model="llama3.2")

# Remote
backend = OllamaBackend(model="llama3.2", host="http://remote-host:11434")

classifier = LLMClassifier(backend)
```

### vLLM

High-throughput serving engine. Constraint mechanism: **`structured_outputs.choice`** (vLLM v0.12.0+, replaces deprecated `guided_choice`; generates bare label text). `score()` uses echo/prefill; `tokenize()` uses forced constrained generation.

**Local server:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 --port 8000
```

**Connect:**
```python
from ollama_classifier.backends import VLLMBackend

backend = VLLMBackend(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:8000/v1",
)
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

Fast serving system for large language models. Constraint mechanism: **regex** (generates bare label text). `score()` uses echo/prefill; `tokenize()` uses forced constrained generation.

**Local server:**
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 --port 30000
```

**Connect:**
```python
from ollama_classifier.backends import SGLangBackend

backend = SGLangBackend(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:30000/v1",
)
```

### llama.cpp

Lightweight inference via `llama-server`. Ideal for CPU or mixed CPU/GPU environments. Constraint mechanism: **GBNF grammar** (generates bare label text). `score()` and `tokenize()` use forced constrained generation (echo not supported by llama.cpp).

**Local server:**
```bash
./llama-server -m model.gguf --host 0.0.0.0 --port 8080 -c 4096
```

**Connect:**
```python
from ollama_classifier.backends import LlamaCppBackend

backend = LlamaCppBackend(
    model="model",
    base_url="http://localhost:8080/v1",
)
```

> **Note:** Logprobs and grammar constraints require llama.cpp to be compiled with the appropriate flags.

### Backend Configuration

All backends share common configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | *(required)* | Model identifier |
| `base_url` | Engine-specific | Base URL of the inference server |
| `api_key` | `"not-needed"` | API key for authentication (not used by `OllamaBackend`) |
| `timeout` | `120.0` | Request timeout in seconds |
| `max_tokens` | `256` | Maximum tokens to generate |
| `extra_body` | `{}` | Extra parameters merged into every request |

`OllamaBackend` additionally accepts a `host` parameter (defaults to `http://localhost:11434`) and accepts pre-initialized `sync_client` / `async_client` instances.

## API Reference

### ClassificationResult

```python
from pydantic import BaseModel

class ClassificationResult(BaseModel):
    prediction: str                          # The predicted class label
    confidence: float                        # Confidence score (0.0 to 1.0)
    probabilities: Dict[str, float]          # Probability distribution over all choices
    method: str = "multi_call"               # "adaptive_generate" or "multi_call"
    approximate: bool = False                # True if any label has partial coverage
    coverage: Dict[str, float] = {}          # Per-label fraction of tokens scored (0.0–1.0)
    n_calls: int = 1                         # Number of API calls made
    raw_response: Dict[str, Any] = {}        # Raw data for debugging
```

### LLMClassifier Methods

| Method | Async | Description |
|--------|-------|-------------|
| `generate(text, choices, system_prompt, *, max_calls)` | `agenerate` | Adaptive constrained generation (1 to `max_calls` calls) |
| `classify(text, choices, system_prompt)` | `aclassify` | Exact multi-call completion scoring (N calls for N labels) |
| `batch_generate(texts, choices, system_prompt, *, max_calls)` | `abatch_generate` | Batch adaptive generation (parallelized) |
| `batch_classify(texts, choices, system_prompt)` | `abatch_classify` | Batch multi-call classification (parallelized) |

### Parameters

- **text** (str): The text to classify
- **texts** (List[str]): List of texts to classify (batch methods)
- **choices** (Union[List[str], Dict[str, str]]): Either a list of choice labels, or a dict mapping labels to descriptions
- **system_prompt** (str | None): Optional custom system prompt
- **max_calls** (int | None): Maximum number of API calls for `generate()`. `1` *(default)* = fast/approximate, `K` = adaptive, `None` = exact

## Sample Data

The package ships with ready-to-use sample datasets under ``examples/sample_data.py`` so you can test the classifier immediately without preparing your own data.

Two datasets are provided, both containing 20 short customer-support ticket texts across 4 labels (`billing`, `technical_support`, `account`, `general`):

- **`DATASET_WITHOUT_DESCRIPTIONS`** — labels as a plain ``list``.
- **`DATASET_WITH_DESCRIPTIONS`** — labels as a ``dict`` mapping each label to a human-readable description, which improves classification accuracy.

### Import the datasets

```python
from examples.sample_data import DATASET_WITHOUT_DESCRIPTIONS, DATASET_WITH_DESCRIPTIONS
```

Each dataset is a ``SampleDataset`` dataclass with these fields:

| Attribute         | Type                            | Description                                      |
|-------------------|---------------------------------|--------------------------------------------------|
| ``texts``         | ``List[str]``                   | Short texts to classify                          |
| ``choices``       | ``List[str]`` or ``Dict[str, str]`` | Label list or label→description dict           |
| ``expected_labels`` | ``List[str]``                 | Expected correct label for each text             |
| ``description``   | ``str``                         | Human-readable description of the dataset        |

### Run a quick test

```python
from ollama_classifier import LLMClassifier
from ollama_classifier.backends import OllamaBackend
from examples.sample_data import DATASET_WITHOUT_DESCRIPTIONS

backend = OllamaBackend(model="llama3.2")
classifier = LLMClassifier(backend)

results = classifier.batch_generate(
    texts=DATASET_WITHOUT_DESCRIPTIONS.texts,
    choices=DATASET_WITHOUT_DESCRIPTIONS.choices,
)

# Verify accuracy
correct = sum(r.prediction == e for r, e in
              zip(results, DATASET_WITHOUT_DESCRIPTIONS.expected_labels))
print(f"Accuracy: {correct}/{len(results)}")
```

### Run the bundled example script

```bash
python -m examples.run_sample_data
```

## License

MIT License

## Development

This project just started! Looking forward to suggestions, issues, and pull requests!
