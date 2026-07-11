"""LOCAL-ONLY integration tests against a real vLLM server.

This file lives under ``local_tests/`` and is NOT part of the CI test suite
(pytest's ``testpaths`` only collects from ``tests/``). It exercises a running
local vLLM instance and the ``Qwen/Qwen2.5-3B-Instruct`` model through both
scoring methods of :class:`LLMClassifier`:

* ``classify()``  -- exact multi-call completion scoring (``method="multi_call"``)
* ``generate()``  -- adaptive constrained generation (``method="adaptive_generate"``)

Prerequisites
-------------
1. vLLM server running on port 30002 with the model loaded:

    docker compose up  # uses the vllm-server service

   Or directly::

    vllm serve Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 30002

2. The model loaded and reachable at ``http://localhost:30002/v1``.

Run with::

    uv run python -m pytest local_tests/test_local_vllm.py -s

The ``-s`` flag disables output capture so the per-test prints are visible.
The whole module is skipped automatically if vLLM is unreachable or the
model is not present, so importing it elsewhere never hard-fails.
"""

from __future__ import annotations

import json
import socket
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

from ollama_classifier import ClassificationResult, LLMClassifier
from ollama_classifier.backends import VLLMBackend

MODEL = "Qwen/Qwen2.5-3B-Instruct"
HOST = "localhost"
PORT = 30002
BASE_URL = f"http://{HOST}:{PORT}/v1"


# ---------------------------------------------------------------------------
# Module-level skip guards: only run against a reachable server + present model
# ---------------------------------------------------------------------------

def _port_open(host: str = HOST, port: int = PORT, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _model_present(model: str = MODEL, host: str = HOST, port: int = PORT) -> bool:
    """Check the vLLM OpenAI-compatible ``/v1/models`` endpoint for the model."""
    try:
        req = Request(f"http://{host}:{port}/v1/models")
        with urlopen(req, timeout=20.0) as resp:
            data = json.loads(resp.read().decode())
    except (URLError, OSError, ValueError):
        return False
    ids = {m.get("id", "") for m in data.get("data", [])}
    return any(model in mid for mid in ids)


_skip_no_server = pytest.mark.skipif(
    not _port_open(), reason=f"vLLM server not reachable at {HOST}:{PORT}"
)
_skip_no_model = pytest.mark.skipif(
    not _model_present(), reason=f"Model '{MODEL}' not found on the vLLM server"
)

pytestmark = [_skip_no_server, _skip_no_model]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier() -> LLMClassifier:
    """Build an ``LLMClassifier`` backed by the local vLLM server.

    Function-scoped to mirror the other backend test files (each async test
    gets its own event loop under ``asyncio_mode=auto``).
    """
    backend = VLLMBackend(model=MODEL, base_url=BASE_URL)
    return LLMClassifier(backend)


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------

def _assert_valid(result: ClassificationResult, choices: list[str], method: str) -> None:
    assert isinstance(result, ClassificationResult)
    assert result.method == method
    assert result.prediction in choices, f"prediction {result.prediction!r} not in choices"
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.probabilities) == set(choices)
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6


# ===========================================================================
# classify() -- exact multi-call completion scoring (N calls for N labels)
# ===========================================================================

class TestClassify:
    def test_classify_basic(self, classifier: LLMClassifier) -> None:
        """Simple 4-way topic classification (multi-call)."""
        text = "The new quantum processor architecture drastically reduces latency."
        choices = ["technology", "sports", "politics", "entertainment"]

        result = classifier.classify(text=text, choices=choices)

        print(f"\n[classify] text={text!r}")
        print(f"  prediction={result.prediction}  confidence={result.confidence:.2%}")
        print(f"  probabilities={result.probabilities}  n_calls={result.n_calls}")

        _assert_valid(result, choices, "multi_call")
        assert result.approximate is False
        assert result.n_calls == len(choices)
        assert result.prediction == "technology"

    def test_classify_with_descriptions(self, classifier: LLMClassifier) -> None:
        """Dict choices mapping labels -> descriptions (multi-call)."""
        text = "This restaurant has amazing food but terrible service."
        choices = {
            "positive": "Text expresses happiness, satisfaction, or approval",
            "negative": "Text expresses anger, disappointment, or disapproval",
            "mixed": "Text contains both positive and negative sentiments",
            "neutral": "Text is factual without strong emotional content",
        }
        labels = list(choices)

        result = classifier.classify(text=text, choices=choices)

        print(f"\n[classify+desc] text={text!r}")
        print(f"  prediction={result.prediction}  confidence={result.confidence:.2%}")
        print(f"  probabilities={result.probabilities}")

        _assert_valid(result, labels, "multi_call")
        assert result.prediction in ("negative", "mixed")

    def test_classify_custom_prompt(self, classifier: LLMClassifier) -> None:
        """Custom system prompt for financial sentiment."""
        text = "The quarterly earnings exceeded analyst expectations."
        choices = ["bullish", "bearish", "neutral"]

        result = classifier.classify(
            text=text,
            choices=choices,
            system_prompt="You are a financial sentiment analyzer. "
            "Classify financial news based on market sentiment.",
        )

        print(f"\n[classify+prompt] text={text!r}")
        print(f"  prediction={result.prediction}  confidence={result.confidence:.2%}")

        _assert_valid(result, choices, "multi_call")
        assert result.prediction == "bullish"


# ===========================================================================
# generate() -- adaptive constrained generation (1..max_calls calls)
# ===========================================================================

class TestGenerate:
    def test_generate_single_call_approximate(self, classifier: LLMClassifier) -> None:
        """max_calls=1 -> single constrained call, fast, possibly approximate."""
        text = "The team won the championship!"
        choices = ["sports", "finance", "science", "politics"]

        result = classifier.generate(text=text, choices=choices, max_calls=1)

        print(f"\n[generate max_calls=1] text={text!r}")
        print(f"  prediction={result.prediction}  confidence={result.confidence:.2%}")
        print(f"  approximate={result.approximate}  n_calls={result.n_calls}")
        print(f"  coverage={result.coverage}")

        _assert_valid(result, choices, "adaptive_generate")
        assert result.n_calls == 1
        assert result.prediction == "sports"

    def test_generate_adaptive_budget(self, classifier: LLMClassifier) -> None:
        """max_calls=3 -> allow up to 3 calls for better cluster resolution."""
        text = "Stock prices plummeted after the announcement."
        choices = ["sports", "finance", "science", "politics"]

        result = classifier.generate(text=text, choices=choices, max_calls=3)

        print(f"\n[generate max_calls=3] text={text!r}")
        print(f"  prediction={result.prediction}  confidence={result.confidence:.2%}")
        print(f"  approximate={result.approximate}  n_calls={result.n_calls}")

        _assert_valid(result, choices, "adaptive_generate")
        assert 1 <= result.n_calls <= 3
        assert result.prediction == "finance"

    def test_generate_exact_unlimited(self, classifier: LLMClassifier) -> None:
        """max_calls=None -> fully recursive resolution (equivalent to exact)."""
        text = "Scientists discovered a new species in the Amazon."
        choices = ["sports", "finance", "science", "politics"]

        result = classifier.generate(text=text, choices=choices, max_calls=None)

        print(f"\n[generate max_calls=None] text={text!r}")
        print(f"  prediction={result.prediction}  confidence={result.confidence:.2%}")
        print(f"  approximate={result.approximate}  n_calls={result.n_calls}")
        print(f"  coverage={result.coverage}")

        _assert_valid(result, choices, "adaptive_generate")
        assert result.prediction in ("science", "politics")


# ===========================================================================
# Async variants
# ===========================================================================

class TestAsync:
    @pytest.mark.asyncio
    async def test_aclassify(self, classifier: LLMClassifier) -> None:
        text = "The concert was an unforgettable experience!"
        choices = ["positive", "negative", "neutral"]

        result = await classifier.aclassify(text=text, choices=choices)

        print(f"\n[aclassify] text={text!r}")
        print(f"  prediction={result.prediction}  confidence={result.confidence:.2%}")

        _assert_valid(result, choices, "multi_call")
        assert result.prediction == "positive"

    @pytest.mark.asyncio
    async def test_agenerate(self, classifier: LLMClassifier) -> None:
        text = "The goalkeeper made an incredible save!"
        choices = ["sports", "finance", "technology"]

        result = await classifier.agenerate(text=text, choices=choices, max_calls=1)

        print(f"\n[agenerate] text={text!r}")
        print(f"  prediction={result.prediction}  confidence={result.confidence:.2%}")

        _assert_valid(result, choices, "adaptive_generate")
        assert result.prediction == "sports"


# ===========================================================================
# Batch variants
# ===========================================================================

class TestBatch:
    def test_batch_classify(self, classifier: LLMClassifier) -> None:
        texts = [
            "The goalkeeper made an incredible save!",
            "The central bank raised interest rates.",
            "The new smartphone features a revolutionary camera.",
        ]
        choices = ["sports", "finance", "technology"]
        expected = ["sports", "finance", "technology"]

        results = classifier.batch_classify(texts=texts, choices=choices)

        assert len(results) == len(texts)
        for text, result, exp in zip(texts, results, expected):
            print(f"\n[batch_classify] {text!r}")
            print(f"  -> {result.prediction} ({result.confidence:.2%})")
            _assert_valid(result, choices, "multi_call")
            assert result.prediction == exp

    def test_batch_generate(self, classifier: LLMClassifier) -> None:
        texts = [
            "The team secured a decisive victory.",
            "Markets rallied on positive economic data.",
            "The software update fixes critical security vulnerabilities.",
        ]
        choices = ["sports", "finance", "technology"]
        expected = ["sports", "finance", "technology"]

        results = classifier.batch_generate(texts=texts, choices=choices, max_calls=1)

        assert len(results) == len(texts)
        for text, result, exp in zip(texts, results, expected):
            print(f"\n[batch_generate] {text!r}")
            print(f"  -> {result.prediction} ({result.confidence:.2%})")
            _assert_valid(result, choices, "adaptive_generate")
            assert result.prediction == exp


# ===========================================================================
# Dataset evaluation -- classify + generate on dataset_runner.py, save CSV
# ===========================================================================

class TestDataset:
    def test_dataset_classify_and_generate(self, classifier: LLMClassifier) -> None:
        """Run the full dataset through classify() and generate(), save CSV."""
        from local_tests.dataset_runner import run_dataset_and_save_csv

        run_dataset_and_save_csv(
            classifier=classifier,
            backend_name="vllm",
            llm_name=MODEL,
        )
