# Scoring & Architecture Redesign Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Eliminate the confidence-concentration bias, consolidate duplicated architecture, and add two principled scoring methods: `classify()` (multi-call completion scoring, length-normalized — gold standard) and `generate()` (adaptive trie-masked generation with `max_calls` budget — exact up to divergence points, recursive cluster resolution).

**Architecture:** Replace the dual-classifier design (`OllamaClassifier` + `LLMClassifier`) with a single `LLMClassifier` backed by a unified `LLMBackend` ABC (including a new `OllamaBackend`). Introduce a `scoring.py` module for all probability math. Two scoring paths:

- **`classify()`** — N completion-scoring calls, geometric-mean normalization. Always exact.
- **`generate()`** — adaptive constrained calls with divergence-aware trie reconstruction. Budget-controlled via `max_calls`. Ranges from 1 call (approximate) to ≤N calls (exact).

**Tech Stack:** Python 3.11+, uv, httpx (required), pydantic (required), pytest, ollama SDK ≥0.12 (optional dependency for `OllamaBackend`)

---

## Honest Comments on Design Decisions

### Decision 1 — Code cleanup & architecture simplification

**Verdict: Correct and overdue.** The 90% duplication between `OllamaClassifier` and `LLMClassifier` is the single biggest architectural liability. Wrapping the Ollama SDK behind an `OllamaBackend(LLMBackend)` and collapsing to one classifier class eliminates ~400 LOC of mirror code and guarantees both scoring methods work identically across all backends. Making `httpx` required is correct — every backend uses it. The `[backends]` optional group creates a false impression that the base package works without HTTP.

### Decision 2 — `classify()` = completion scoring (3.3) with geometric-mean normalization (3.1)

**Verdict: Gold-standard method.** Scoring a given completion (without forcing generation through constrained decoding) avoids all constraint-dependent logprob semantics (pre-mask vs post-mask). Geometric-mean normalization ($\frac{1}{|tokens|}\sum \log P$) is the textbook fix for length bias and is provably length-invariant.

The cost is N API calls per classification, but async `aclassify()` parallelizes via `asyncio.gather`, and sync `batch_classify` uses `ThreadPoolExecutor`. Real-world latency is bounded by the slowest single call, not N× latency.

**Consistency note:** All confidence calculations across both methods use **geometric mean** (not arithmetic mean) for length normalization. This is applied uniformly in `scoring.length_normalize()`.

### Decision 3 — `generate()` = adaptive trie-masked generation with `max_calls`

**Verdict: Elegant, with a principled solution to the approximation problem.**

#### The divergence-aware strategy

The key insight: **when computing logprobs for each label, only consider tokens up to and including the divergence point from the winning path.**

For a winning path $W$ and a label $L$, the **divergence point** $d(L, W)$ is:

$$d(L, W) = \min\{i : l_i \neq w_i\}$$

For all positions $i \leq d$, the model's distribution at position $i$ is conditioned on the prefix $[w_0, \dots, w_{i-1}] = [l_0, \dots, l_{i-1}]$ — the **same prefix** for both paths. At position $d$ specifically, both $w_d$ and $l_d$ are siblings under the same trie node, so $l_d$'s logprob appears in `top_logprobs[d]` and is **exact**.

At positions $i > d$, the conditioning context has diverged — $l_i$'s logprob would be conditioned on the winner's tokens, not the label's own tokens. These positions are **excluded** from scoring.

#### Solving Case 2: recursive cluster resolution

When labels diverge from the winner at the same point and share prefixes among themselves, they form **unresolved clusters**. Each cluster is a sub-trie hanging off the winning path at a branch point.

For each unresolved cluster, we make **one supplementary constrained call** restricted to that cluster's labels. The model walks one path through the sub-trie. The cluster's winner is fully resolved. Other cluster members are resolved up to their divergence from the cluster winner. If sub-clusters remain unresolved, the algorithm recurses.

**The `max_calls` parameter controls the budget:**

```python
def generate(
    self,
    text: str,
    choices: ChoicesType,
    system_prompt: str | None = None,
    *,
    max_calls: int | None = 1,
) -> ClassificationResult:
```

| Value | Behavior | Calls | Exactness |
|---|---|---|---|
| `max_calls=1` (default) | Single call, partial scoring up to divergence | 1 | Partial (approximate) |
| `max_calls=K` | Up to K calls, adaptive cluster resolution | 1 to K | Improves with each call |
| `max_calls=None` | Resolve everything recursively | 1 to N | Fully exact |

#### When is `generate()` exact without approximation?

For pre-mask backends (all four supported backends — see Section: Backend Constraint Analysis), `generate(max_calls=None)` is **fully exact**: every token of every label is scored under a conditioning prefix that matches the label's own prefix up to that point. This is because:

1. **Shared-prefix tokens** ($i < d$): conditioned on the same prefix by definition.
2. **Divergence token** ($i = d$): siblings under the same trie node, same conditioning.
3. **Supplementary call tokens**: the cluster call starts from the same prompt, and the sub-trie ensures the model walks the cluster's shared prefix before branching.

#### What the `approximate` flag means

The `ClassificationResult` carries an `approximate: bool` field. It is `True` when **any** label has unresolved tokens (scored on fewer tokens than its full length). It is `False` when every label is fully resolved (either `max_calls=None` with enough calls, or all labels share long enough prefixes that a single call resolves everything).

A `coverage: dict[str, float]` field reports, for each label, the fraction of tokens that were scored (e.g., `{"positive": 1.0, "negative": 1.0, "technical_support": 0.25}`).

#### Call complexity

Let $N$ = number of labels, $B$ = average branching factor of the trie.

| Label structure | Calls (max_calls=None) | Example |
|---|---|---|
| All share one prefix, branch at the end | **1** | `["tech_support", "tech_issues"]` |
| Two prefix clusters | **2** | `["tech_support", "tech_issues", "billing", "general"]` |
| Worst case: all diverge at token 0 | **N** | Equal to `classify()` |

In practice, classification labels cluster hierarchically, so the typical case is $O(\log N)$ to $O(N/B)$ calls — **significantly fewer than N**.

---

## Backend Constraint Analysis

The `generate()` method requires backends to: (1) constrain output to valid labels, and (2) return **pre-mask** logprobs (model's raw distribution before constraint masking). Here is the exact status per backend:

| Backend | Constraint mechanism | Generates bare label? | Logprobs pre-mask? | Tokenization risk |
|---|---|---|---|---|
| **vLLM** | `guided_choice` (native) | ✅ | ✅ | None |
| **SGLang** | `regex` (bare labels) | ✅ | ✅ | None |
| **Ollama** (≥0.12) | JSON Schema `enum` via `format` | ❌ (JSON wrapper) | ✅ (v0.12.11+) | **High** — context-sensitive |
| **llama.cpp** | GBNF `grammar` | ✅ | ✅ | None |

### vLLM — cleanest path

vLLM supports `guided_choice` natively via `extra_body`:

```python
body["guided_choice"] = ["positive", "negative", "neutral"]
```

The model generates **bare label text** with no JSON wrapper. Logprobs contain only label tokens. Trie reconstruction is straightforward. vLLM returns pre-mask logprobs (raw model logits before guided decoding masking).

### SGLang — regex constraint

SGLang supports regex constraints for bare-label generation:

```python
body["regex"] = r"(positive|negative|neutral)"
```

The model generates bare label text. Logprobs are pre-mask. Clean reconstruction.

### Ollama (≥0.12) — JSON Schema with context-dependent tokenization

**What works:**
- ✅ JSON Schema `format` with `enum` constraint — confirmed working
- ✅ Native logprobs (`/api/chat`, `/api/generate`) as of v0.12.11, returning pre-mask logits

**What doesn't work:**
- ❌ No native `guided_choice` parameter. The API only accepts JSON Schema via `format`.

**The context-sensitive tokenization problem:**

Ollama's grammar engine (llama.cpp's GBNF under the hood) constrains at the character/token level. When you pass:

```json
{"label": {"type": "string", "enum": ["positive", "negative"]}}
```

The model is forced to generate: `{` `"label"` `:` `"` `positive` `"` `}`

The logprobs returned cover **all** these tokens. To reconstruct label scores, you must filter out the structural tokens and keep only the label tokens. **But** the label token sequence inside the JSON wrapper may differ from standalone tokenization:

- `tokenize("positive")` in isolation → `["positive"]` (1 token)
- `tokenize("positive")` preceded by `"` in the JSON → could be `["positive"]` or `["\"positive"]` depending on BPE merge rules

**Solution:** The `OllamaBackend.tokenize()` method tokenizes labels **in the JSON context they actually appear in**:

```python
def tokenize(self, text: str, *, context: str | None = None) -> list[Token]:
    # For generate(), context will be the JSON prefix: '{"label": "'
    # so the label is tokenized exactly as it appears in the response
    full_text = (context or "") + text
    response = client.tokenize(model=self._model, text=full_text)
    tokens = response.get("tokens", [])
    if context:
        ctx_response = client.tokenize(model=self._model, text=context)
        tokens = tokens[len(ctx_response["tokens"]):]
    return [Token(text="", id=t) for t in tokens]
```

The classifier passes the JSON prefix as context when calling `tokenize()`, so the trie is built from **exactly the tokens that will appear in the response**. The `_extract_step_logprobs()` method then filters `top_logprobs` against this token set, ignoring structural JSON tokens.

**Additional complexity for cluster calls:** When a supplementary call targets a sub-cluster of labels, the Ollama backend must pass a JSON schema with only the sub-cluster's labels in the enum. The JSON prefix is the same (`{"label": "`), so context-dependent tokenization remains consistent.

### llama.cpp — GBNF grammar for bare labels

**What works:**
- ✅ GBNF grammar constraints — native and reliable
- ✅ Logprobs — pre-mask (model's raw distribution before grammar masking)
- ✅ GBNF can express bare label alternatives directly

**What doesn't work:**
- ❌ `response_format` with JSON schema on `/v1/chat/completions` is **buggy** (GitHub issues #11988, #11847)
- ❌ No `guided_choice` parameter

**Solution:** Use GBNF grammar on the chat completions endpoint. `llama-server` accepts a non-standard `grammar` field:

```python
body["grammar"] = 'root ::= "positive" | "negative" | "neutral"'
```

This generates **bare label text** — no JSON wrapper. The logprobs stream contains only label tokens. This is cleaner than Ollama's JSON approach.

For supplementary cluster calls, generate a GBNF grammar with only the cluster's labels.

### Backend abstraction

The `LLMBackend` ABC exposes a high-level `constrain_to_labels()` method that each backend implements according to its capabilities:

```python
class LLMBackend(ABC):
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        constrain_labels: list[str] | None = None,  # replaces guided_choice/guided_json
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> ChatResponse:
        """Constrained generation. Each backend translates constrain_labels
        to its native constraint mechanism (guided_choice, regex, JSON enum, GBNF)."""
        ...
```

Each backend translates `constrain_labels` internally:
- **vLLM**: `guided_choice=labels`
- **SGLang**: `regex=build_alternation(labels)`
- **Ollama**: `format=build_json_enum(labels)`
- **llama.cpp**: `grammar=build_gbnf(labels)`

The `tokenize()` method receives the appropriate `context` so the trie matches the actual response tokens.

---

## Architecture Overview

### New file structure

```
src/ollama_classifier/
├── __init__.py            # Public API: LLMClassifier, ClassificationResult, backends
├── types.py               # ClassificationResult (Pydantic), ChoicesType
├── prompts.py             # Prompt building (unchanged logic)
├── scoring.py             # NEW: softmax, geometric-mean norm, trie, divergence-aware scoring
├── classifier.py          # REWRITE: single LLMClassifier class
└── backends/
    ├── __init__.py        # Exports all backends
    ├── base.py            # REWRITE: enhanced LLMBackend ABC with constrain_labels
    ├── ollama.py          # NEW: OllamaBackend (ollama SDK ≥0.12)
    ├── vllm.py            # REWRITE: enhanced VLLMBackend
    ├── sglang.py          # REWRITE: enhanced SGLangBackend
    └── llamacpp.py        # REWRITE: enhanced LlamaCppBackend
```

### Deleted files

- `src/ollama_classifier/llm_classifier.py` — merged into `classifier.py`
- `src/ollama_classifier/classifier.py` (old `OllamaClassifier`) — replaced

### Public API

```python
class LLMClassifier:
    def __init__(self, backend: LLMBackend, *, max_workers: int = 4): ...

    # Adaptive constrained generation with divergence-aware confidence
    def generate(self, text, choices, system_prompt=None, *,
                 max_calls: int | None = 1) -> ClassificationResult: ...
    async def agenerate(self, text, choices, system_prompt=None, *,
                        max_calls: int | None = 1) -> ClassificationResult: ...

    # Multi-call completion scoring with geometric-mean confidence (exact)
    def classify(self, text, choices, system_prompt=None) -> ClassificationResult: ...
    async def aclassify(self, text, choices, system_prompt=None) -> ClassificationResult: ...

    # Batch (parallelized via ThreadPoolExecutor / asyncio.gather)
    def batch_generate(self, texts, choices, system_prompt=None, *,
                       max_calls: int | None = 1) -> list[ClassificationResult]: ...
    def batch_classify(self, texts, choices, system_prompt=None) -> list[ClassificationResult]: ...
    async def abatch_generate(...) -> list[ClassificationResult]: ...
    async def abatch_classify(...) -> list[ClassificationResult]: ...
```

**Breaking change:** `generate()` now returns `ClassificationResult` (not `str`). Users who want only the label access `.prediction`. Documented in changelog.

### Backend interface (enhanced)

```python
@dataclass
class TokenLogprob:
    """A single token with its log probability and alternatives."""
    token: str
    token_id: int = -1
    logprob: float = 0.0
    top_logprobs: list[dict[str, float]] = field(default_factory=list)

@dataclass
class ChatResponse:
    """Response from a constrained generation call."""
    content: str
    label: str = ""
    logprobs: list[TokenLogprob] | None = None
    raw: dict = field(default_factory=dict)

@dataclass
class ScoringResponse:
    """Response from a completion-scoring call (for classify())."""
    completion: str
    logprobs: list[TokenLogprob] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

@dataclass
class Token:
    """A tokenized unit."""
    text: str       # detokenized text of this token
    id: int         # token ID in the model's vocabulary

class LLMBackend(ABC):
    def chat(
        self, messages, *,
        temperature=0.0,
        constrain_labels: list[str] | None = None,  # high-level: constrain to these labels
        logprobs=False, top_logprobs=5,
    ) -> ChatResponse:
        """Each backend translates constrain_labels to its native mechanism."""

    async def achat(...) -> ChatResponse: ...

    def score(
        self, messages: list[ChatMessage], completion: str,
    ) -> ScoringResponse:
        """Score a completion's per-token logprobs without generation."""

    async def ascore(...) -> ScoringResponse: ...

    def tokenize(
        self, text: str, *, context: str | None = None,
    ) -> list[Token]:
        """Tokenize text. If context given, tokenize context+text and return
        only tokens for text (context-dependent tokenization)."""

    async def atokenize(...) -> list[Token]: ...

    # Backend capability flags (for documentation/testing)
    @property
    @abstractmethod
    def supports_bare_label_constraint(self) -> bool:
        """True if chat() generates bare label text (no JSON wrapper).
        vLLM/SGLang/llama.cpp: True. Ollama: False."""
```

---

## Task Breakdown

### Task 1: Create scoring module — probability functions

**Objective:** Implement geometric-mean normalization and stable softmax as pure functions.

**Files:**
- Create: `src/ollama_classifier/scoring.py`
- Create: `tests/test_scoring.py`

**Step 1: Write failing tests**

```python
# tests/test_scoring.py
import math
from ollama_classifier.scoring import geometric_mean_logprob, stable_softmax

def test_geometric_mean_single_token():
    assert geometric_mean_logprob([-0.5]) == -0.5

def test_geometric_mean_multi_token():
    lps = [math.log(0.9)] * 4
    expected = sum(lps) / 4  # = log(0.9) ≈ -0.1054
    assert abs(geometric_mean_logprob(lps) - expected) < 1e-10

def test_geometric_mean_removes_length_bias():
    """Labels with identical per-token confidence get identical scores."""
    short = [math.log(0.95)]         # 1 token
    long_ = [math.log(0.95)] * 4     # 4 tokens
    assert abs(geometric_mean_logprob(short) - geometric_mean_logprob(long_)) < 1e-10

def test_geometric_mean_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        geometric_mean_logprob([])

def test_softmax_basic():
    probs = stable_softmax({"a": -0.1, "b": -2.0, "c": -5.0})
    assert abs(sum(probs.values()) - 1.0) < 1e-10
    assert probs["a"] > probs["b"] > probs["c"]

def test_softmax_stability():
    probs = stable_softmax({"short": -0.02, "long": -15.0})
    assert probs["short"] > probs["long"]
    assert probs["short"] < 1.0  # not catastrophically concentrated

def test_softmax_handles_inf():
    probs = stable_softmax({"a": -1.0, "b": float("-inf")})
    assert probs["a"] == 1.0
    assert probs["b"] == 0.0

def test_softmax_all_inf_returns_uniform():
    probs = stable_softmax({"a": float("-inf"), "b": float("-inf")})
    assert probs["a"] == 0.5
    assert probs["b"] == 0.5
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/test_scoring.py -v
# Expected: FAIL — ModuleNotFoundError
```

**Step 3: Implement scoring functions**

```python
# src/ollama_classifier/scoring.py
"""Probability and scoring utilities for classification.

All length normalization uses geometric mean (not arithmetic mean),
applied consistently across both generate() and classify() methods.
"""
import math
from typing import Dict, Sequence


def geometric_mean_logprob(logprobs: Sequence[float]) -> float:
    """Compute the geometric-mean (length-normalized) log probability.

    This is the per-token average of log probabilities:
        score = (1/N) * Σ log P(token_i)

    Equivalent to log of the geometric mean of token probabilities.
    Eliminates the length bias that occurs when summing raw logprobs
    over labels with different token counts.

    Args:
        logprobs: Per-token log probabilities (must be non-empty).

    Returns:
        Geometric-mean log probability score.

    Raises:
        ValueError: If logprobs is empty.
    """
    if not logprobs:
        raise ValueError("Cannot compute geometric mean of empty sequence.")
    valid = [lp for lp in logprobs if lp > float("-inf")]
    if not valid:
        return float("-inf")
    return sum(valid) / len(valid)


def stable_softmax(logprobs: Dict[str, float]) -> Dict[str, float]:
    """Numerically stable softmax over a dictionary of log probabilities.

    Args:
        logprobs: Dict mapping labels to log probability scores.

    Returns:
        Dict mapping labels to probabilities (summing to 1.0).
    """
    valid = {k: v for k, v in logprobs.items() if v > float("-inf")}

    if not valid:
        n = len(logprobs)
        return {k: 1.0 / n for k in logprobs}

    max_lp = max(valid.values())
    exp_vals = {
        k: math.exp(v - max_lp) if v > float("-inf") else 0.0
        for k, v in logprobs.items()
    }
    total = sum(exp_vals.values())

    if total == 0:
        n = len(logprobs)
        return {k: 1.0 / n for k in logprobs}

    return {k: v / total for k, v in exp_vals.items()}
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_scoring.py -v
# Expected: 8 passed
```

**Step 5: Commit**

```bash
git add src/ollama_classifier/scoring.py tests/test_scoring.py
git commit -m "feat: add scoring module with geometric-mean normalization and stable softmax"
```

---

### Task 2: Add trie data structure to scoring module

**Objective:** Build a prefix trie from label token-sequences, with branching factor computation.

**Files:**
- Modify: `src/ollama_classifier/scoring.py`
- Modify: `tests/test_scoring.py`

**Step 1: Write failing tests**

```python
# Append to tests/test_scoring.py
from ollama_classifier.scoring import LabelTrie, TrieNode

def test_trie_single_token_labels():
    trie = LabelTrie()
    trie.insert("positive", ["positive"])
    trie.insert("negative", ["negative"])
    trie.insert("neutral", ["neutral"])
    assert len(trie.root.children) == 3
    assert trie.root.children["positive"].is_terminal
    assert trie.root.children["positive"].label == "positive"

def test_trie_shared_prefix():
    trie = LabelTrie()
    trie.insert("account", ["acc", "ount"])
    trie.insert("access", ["acc", "ess"])
    assert len(trie.root.children) == 1
    acc = trie.root.children["acc"]
    assert not acc.is_terminal
    assert len(acc.children) == 2

def test_trie_max_branching_factor():
    trie = LabelTrie()
    trie.insert("positive", ["positive"])
    trie.insert("negative", ["negative"])
    trie.insert("neutral", ["neutral"])
    trie.insert("account", ["acc", "ount"])
    trie.insert("access", ["acc", "ess"])
    assert trie.max_branching_factor == 3

def test_trie_get_token_sequence():
    trie = LabelTrie()
    trie.insert("technical_support", ["techn", "ical", "_support"])
    assert trie.get_token_sequence("technical_support") == ["techn", "ical", "_support"]
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/test_scoring.py::test_trie_single_token_labels -v
# Expected: FAIL — ImportError
```

**Step 3: Implement trie**

```python
# Append to src/ollama_classifier/scoring.py
from dataclasses import dataclass, field


@dataclass
class TrieNode:
    """A node in the label prefix trie."""
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    is_terminal: bool = False
    label: str | None = None


class LabelTrie:
    """Prefix trie over label token sequences.

    Used by generate() to:
    1. Determine the minimum top_logprobs K (max branching factor).
    2. Find divergence points between the winning path and each label.
    3. Identify unresolved clusters for recursive resolution.
    """

    def __init__(self) -> None:
        self.root: TrieNode = TrieNode()
        self._token_sequences: dict[str, list[str]] = {}

    def insert(self, label: str, tokens: list[str]) -> None:
        self._token_sequences[label] = tokens
        node = self.root
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_terminal = True
        node.label = label

    @property
    def max_branching_factor(self) -> int:
        """Maximum number of children at any node. This is the minimum
        top_logprobs K needed to capture all sibling alternatives."""
        return self._max_branching(self.root)

    def _max_branching(self, node: TrieNode) -> int:
        if not node.children:
            return 0
        child_max = max(self._max_branching(c) for c in node.children.values())
        return max(len(node.children), child_max)

    def get_token_sequence(self, label: str) -> list[str]:
        return self._token_sequences[label]

    def all_labels(self) -> list[str]:
        return list(self._token_sequences.keys())
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_scoring.py -v
# Expected: 12 passed
```

**Step 5: Commit**

```bash
git add src/ollama_classifier/scoring.py tests/test_scoring.py
git commit -m "feat: add label prefix trie for adaptive scoring"
```

---

### Task 3: Add divergence-aware scoring and cluster resolution

**Objective:** Implement the core algorithm — divergence-point computation, partial scoring up to divergence, cluster identification, and recursive resolution.

**Files:**
- Modify: `src/ollama_classifier/scoring.py`
- Modify: `tests/test_scoring.py`

**Step 1: Write failing tests**

```python
# Append to tests/test_scoring.py
from ollama_classifier.scoring import (
    divergence_point,
    score_labels_from_winning_path,
    identify_unresolved_clusters,
    Cluster,
)


def test_divergence_point_identical():
    assert divergence_point(["a", "b", "c"], ["a", "b", "c"]) == 3  # full match

def test_divergence_point_first_token():
    assert divergence_point(["x", "b"], ["a", "b"]) == 0

def test_divergence_point_middle():
    assert divergence_point(["a", "x", "c"], ["a", "b", "c"]) == 1

def test_divergence_point_different_lengths():
    # Shorter label is prefix of longer
    assert divergence_point(["a", "b"], ["a", "b", "c"]) == 2  # diverges at position 2 (one ends)


def test_score_labels_case1_winner_a():
    """Case 1 from the design discussion: winner = a, b diverges at token 3, c at token 0.
    All labels should be scored up to their divergence point."""
    token_seqs = {
        "a": ["t1", "t2", "t3", "t4a"],
        "b": ["t1", "t2", "t3", "t4b"],
        "c": ["t1c", "t2c", "t3c", "t4c"],
    }
    # Simulate top_logprobs from the winning path (a)
    step_logprobs = [
        {"t1": -0.1, "t1c": -2.5, "other": -5.0},   # pos 0: a,b share t1; c uses t1c
        {"t2": -0.2, "x": -3.0},                      # pos 1
        {"t3": -0.15, "y": -2.8},                     # pos 2
        {"t4a": -0.1, "t4b": -1.5, "z": -4.0},        # pos 3: a vs b divergence
    ]

    scores = score_labels_from_winning_path(token_seqs, "a", step_logprobs)

    # a: all 4 tokens, geometric mean
    assert abs(scores["a"] - ((-0.1 + -0.2 + -0.15 + -0.1) / 4)) < 1e-10
    # b: all 4 tokens (t1, t2, t3 shared + t4b at divergence), geometric mean
    assert abs(scores["b"] - ((-0.1 + -0.2 + -0.15 + -1.5) / 4)) < 1e-10
    # c: only 1 token (t1c at pos 0)
    assert abs(scores["c"] - (-2.5)) < 1e-10


def test_score_labels_case2_winner_c():
    """Case 2: winner = c, a and b diverge at token 0 but share prefix among themselves.
    a and b get same score (only token 0 resolved)."""
    token_seqs = {
        "a": ["t1", "t2", "t3", "t4a"],
        "b": ["t1", "t2", "t3", "t4b"],
        "c": ["t1c", "t2c", "t3c", "t4c"],
    }
    step_logprobs = [
        {"t1": -2.0, "t1c": -0.1, "other": -5.0},   # pos 0: c wins
        {"t2c": -0.2, "x": -3.0},
        {"t3c": -0.15, "y": -2.8},
        {"t4c": -0.1, "z": -4.0},
    ]

    scores = score_labels_from_winning_path(token_seqs, "c", step_logprobs)
    # a and b: only token 0 (t1), same score
    assert abs(scores["a"] - (-2.0)) < 1e-10
    assert abs(scores["b"] - (-2.0)) < 1e-10
    # c: all 4 tokens
    assert abs(scores["c"] - ((-0.1 + -0.2 + -0.15 + -0.1) / 4)) < 1e-10


def test_identify_unresolved_clusters_case2():
    """After scoring case 2 (winner=c), a and b form an unresolved cluster."""
    token_seqs = {
        "a": ["t1", "t2", "t3", "t4a"],
        "b": ["t1", "t2", "t3", "t4b"],
        "c": ["t1c", "t2c", "t3c", "t4c"],
    }
    scored_lengths = {"a": 1, "b": 1, "c": 4}  # how many tokens resolved per label

    clusters = identify_unresolved_clusters(token_seqs, scored_lengths)
    assert len(clusters) == 1
    assert set(clusters[0].labels) == {"a", "b"}
    assert clusters[0].resolved_length == 1  # they share 1 token already


def test_identify_no_clusters_when_all_resolved():
    token_seqs = {"a": ["t1"], "b": ["t2"]}
    scored_lengths = {"a": 1, "b": 1}
    clusters = identify_unresolved_clusters(token_seqs, scored_lengths)
    assert len(clusters) == 0
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/test_scoring.py::test_divergence_point_identical -v
# Expected: FAIL — ImportError
```

**Step 3: Implement divergence-aware scoring**

```python
# Append to src/ollama_classifier/scoring.py
from dataclasses import dataclass


@dataclass
class Cluster:
    """A group of labels that share a token prefix and need further resolution."""
    labels: list[str]
    resolved_length: int  # how many tokens are already scored


def divergence_point(label_tokens: list[str], winning_tokens: list[str]) -> int:
    """Find the first position where label_tokens and winning_tokens differ.

    Returns:
        Index of first divergence. If identical, returns len(label_tokens).
        If label is shorter (prefix of winner), returns len(label_tokens).
    """
    min_len = min(len(label_tokens), len(winning_tokens))
    for i in range(min_len):
        if label_tokens[i] != winning_tokens[i]:
            return i
    return min_len  # no divergence within overlap; label is prefix or equal


def score_labels_from_winning_path(
    token_sequences: dict[str, list[str]],
    winning_label: str,
    step_logprobs: list[dict[str, float]],
) -> dict[str, float]:
    """Score all labels using divergence-aware partial scoring.

    For each label, computes the geometric-mean logprob over tokens [0, d]
    where d is the divergence point from the winning path.

    Tokens at positions 0..d are exact because the conditioning prefix
    matches for both the label and the winner up to that point.

    Args:
        token_sequences: {label: [token strings]} for all labels.
        winning_label: The label that the model actually generated.
        step_logprobs: Per-step top_logprobs from the constrained call.
            step_logprobs[i] = {token_str: logprob} for position i.

    Returns:
        {label: geometric_mean_logprob} for each label.
    """
    winning_tokens = token_sequences[winning_label]
    scores: dict[str, float] = {}

    for label, label_tokens in token_sequences.items():
        d = divergence_point(label_tokens, winning_tokens)
        # Score tokens at positions 0..d (inclusive)
        # d+1 tokens total, but cap at available step_logprobs
        n_scoring = min(d + 1, len(label_tokens), len(step_logprobs))
        token_lps: list[float] = []

        for i in range(n_scoring):
            token = label_tokens[i]
            if i < len(step_logprobs) and token in step_logprobs[i]:
                token_lps.append(step_logprobs[i][token])
            else:
                token_lps.append(float("-inf"))

        scores[label] = geometric_mean_logprob(token_lps) if token_lps else float("-inf")

    return scores


def get_scored_lengths(
    token_sequences: dict[str, list[str]],
    winning_label: str,
) -> dict[str, int]:
    """Get the number of tokens scored per label (divergence_point + 1, capped)."""
    winning_tokens = token_sequences[winning_label]
    lengths: dict[str, int] = {}
    for label, label_tokens in token_sequences.items():
        d = divergence_point(label_tokens, winning_tokens)
        n = min(d + 1, len(label_tokens))
        lengths[label] = n
    return lengths


def identify_unresolved_clusters(
    token_sequences: dict[str, list[str]],
    scored_lengths: dict[str, int],
) -> list[Cluster]:
    """Identify groups of labels that are not fully resolved and share prefixes.

    Two labels are in the same cluster if:
    1. Both have scored_length < their full token length (unresolved), AND
    2. They share the same tokens at positions [0, min(scored_lengths)-1].

    Each cluster records the deepest shared prefix length already resolved.
    """
    unresolved = {
        label: seq
        for label, seq in token_sequences.items()
        if scored_lengths.get(label, 0) < len(seq)
    }

    if not unresolved:
        return []

    # Group by prefix at the already-scored length
    clusters: dict[tuple[str, ...], list[str]] = {}
    for label, seq in unresolved.items():
        resolved = scored_lengths.get(label, 0)
        # The shared prefix that was already scored
        prefix = tuple(seq[:resolved])
        clusters.setdefault(prefix, []).append(label)

    return [
        Cluster(labels=labels, resolved_length=len(prefix))
        for prefix, labels in clusters.items()
        if len(labels) > 0
    ]
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_scoring.py -v
# Expected: 18 passed
```

**Step 5: Commit**

```bash
git add src/ollama_classifier/scoring.py tests/test_scoring.py
git commit -m "feat: add divergence-aware scoring and recursive cluster resolution"
```

---

### Task 4: Redesign types module

**Objective:** Migrate `ClassificationResult` to Pydantic, add `method`, `approximate`, and `coverage` fields.

**Files:**
- Modify: `src/ollama_classifier/types.py`

```python
# src/ollama_classifier/types.py
"""Type definitions for ollama-classifier."""
from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    """Result of a classification operation.

    Attributes:
        prediction: The predicted class label.
        confidence: Confidence score for the prediction (0.0 to 1.0).
        probabilities: Probability distribution over all choices.
        method: Scoring method used: "adaptive_generate" or "multi_call".
        approximate: True if any label has partial coverage (unresolved tokens).
        coverage: Per-label fraction of tokens scored (0.0 to 1.0).
            1.0 = fully resolved. Only present for adaptive_generate.
        n_calls: Number of API calls made.
        raw_response: Raw data for debugging.
    """
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    method: str = "multi_call"
    approximate: bool = False
    coverage: Dict[str, float] = Field(default_factory=dict)
    n_calls: int = 1
    raw_response: Dict[str, Any] = Field(default_factory=dict)


ChoicesType = Union[List[str], Dict[str, str]]
```

**Commit:**

```bash
git add src/ollama_classifier/types.py
git commit -m "refactor: migrate ClassificationResult to Pydantic with method/coverage fields"
```

---

### Task 5: Redesign backend base class

**Objective:** Enhance `LLMBackend` ABC with `constrain_labels`, `score()`, `tokenize()`, and the `supports_bare_label_constraint` capability flag.

**Files:**
- Modify: `src/ollama_classifier/backends/base.py`

```python
# src/ollama_classifier/backends/base.py
"""Base backend protocol for inference engines."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class TokenLogprob:
    token: str
    token_id: int = -1
    logprob: float = 0.0
    top_logprobs: list[dict[str, float]] = field(default_factory=list)


@dataclass
class ChatResponse:
    content: str
    label: str = ""
    logprobs: Optional[List[TokenLogprob]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringResponse:
    completion: str
    logprobs: List[TokenLogprob] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Token:
    text: str
    id: int


class LLMBackend(ABC):
    """Abstract base class for LLM inference backends."""

    def __init__(
        self,
        model: str,
        base_url: str = "",
        *,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_tokens: int = 256,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._api_key = api_key or "not-needed"
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._extra_body = extra_body or {}

    @property
    def model(self) -> str:
        return self._model

    @property
    @abstractmethod
    def supports_bare_label_constraint(self) -> bool:
        """True if chat() generates bare label text (no JSON wrapper).
        vLLM/SGLang/llama.cpp: True. Ollama: False (uses JSON enum)."""

    # --- Sync ---

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        constrain_labels: Optional[List[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> ChatResponse:
        """Constrained generation. Each backend translates constrain_labels
        to its native constraint mechanism."""

    @abstractmethod
    def score(self, messages: List[ChatMessage], completion: str) -> ScoringResponse:
        """Score a completion by computing per-token logprobs of the
        completion text given the message context. No generation occurs."""

    @abstractmethod
    def tokenize(self, text: str, *, context: Optional[str] = None) -> List[Token]:
        """Tokenize text. If context provided, tokenize context+text and
        return only tokens for text (context-dependent tokenization)."""

    # --- Async ---

    @abstractmethod
    async def achat(self, messages, **kwargs) -> ChatResponse: ...

    @abstractmethod
    async def ascore(self, messages, completion: str) -> ScoringResponse: ...

    @abstractmethod
    async def atokenize(self, text, *, context=None) -> List[Token]: ...

    # --- Shared HTTP helpers (for OpenAI-compatible backends) ---

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _build_chat_body(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        constrain_labels: Optional[List[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> Dict[str, Any]:
        """Build request body for OpenAI-compatible /v1/chat/completions.
        Subclasses override _apply_constraint() to add backend-specific fields."""
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if constrain_labels is not None:
            self._apply_constraint(body, constrain_labels)
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        body.update(self._extra_body)
        return body

    def _apply_constraint(self, body: Dict[str, Any], labels: List[str]) -> None:
        """Apply backend-specific output constraint. Override in subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support label constraints"
        )

    @staticmethod
    def _parse_chat_response(data: Dict[str, Any]) -> ChatResponse:
        choice = data["choices"][0]
        content = choice["message"].get("content", "")

        logprobs_list: Optional[List[TokenLogprob]] = None
        if choice.get("logprobs") and choice["logprobs"].get("content"):
            logprobs_list = []
            for ti in choice["logprobs"]["content"]:
                top_lps = {}
                for alt in ti.get("top_logprobs", []):
                    top_lps[alt["token"]] = alt["logprob"]
                logprobs_list.append(TokenLogprob(
                    token=ti["token"],
                    logprob=ti["logprob"],
                    top_logprobs=top_lps,
                ))

        return ChatResponse(content=content, logprobs=logprobs_list, raw=data)
```

**Commit:**

```bash
git add src/ollama_classifier/backends/base.py
git commit -m "refactor: redesign LLMBackend ABC with constrain_labels, score, tokenize"
```

---

### Task 6: Implement OllamaBackend

**Objective:** Wrap the Ollama SDK (≥0.12) behind the `LLMBackend` interface, implementing all methods with JSON Schema enum constraints and context-dependent tokenization.

**Files:**
- Create: `src/ollama_classifier/backends/ollama.py`

```python
# src/ollama_classifier/backends/ollama.py
"""Ollama inference backend (requires Ollama ≥0.12 for logprobs support).

Wraps the Ollama Python SDK behind the LLMBackend interface.

Constraint mechanism: JSON Schema enum via the `format` parameter.
The model generates JSON: {"label": "<chosen_label>"}. Structural JSON
tokens ({, "label", :, ", }) must be filtered during trie reconstruction.
Context-dependent tokenization is used so the trie matches the actual
response tokens.
"""
import json
from typing import Any, Dict, List, Optional

from .base import ChatMessage, ChatResponse, LLMBackend, ScoringResponse, Token, TokenLogprob


class OllamaBackend(LLMBackend):
    """Backend for the Ollama runtime (≥v0.12) via the official Python SDK."""

    def __init__(
        self,
        model: str,
        *,
        host: Optional[str] = None,
        sync_client: Any = None,
        async_client: Any = None,
        timeout: float = 120.0,
        max_tokens: int = 256,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model=model,
            base_url=host or "http://localhost:11434",
            timeout=timeout,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        self._sync_client = sync_client
        self._async_client = async_client

    @property
    def supports_bare_label_constraint(self) -> bool:
        return False  # Ollama uses JSON enum wrapper

    # --- Client management ---

    def _get_sync_client(self):
        if self._sync_client is None:
            from ollama import Client
            self._sync_client = Client(host=self._base_url, timeout=self._timeout)
        return self._sync_client

    async def _get_async_client(self):
        if self._async_client is None:
            from ollama import AsyncClient
            self._async_client = AsyncClient(host=self._base_url, timeout=self._timeout)
        return self._async_client

    # --- Constraint translation ---

    @staticmethod
    def _build_json_enum(labels: List[str]) -> Dict[str, Any]:
        """Build JSON schema with enum constraint for Ollama's format parameter."""
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": labels},
            },
            "required": ["label"],
        }

    def _apply_constraint(self, body: Dict[str, Any], labels: List[str]) -> None:
        body["format"] = self._build_json_enum(labels)

    @staticmethod
    def _json_label_context() -> str:
        """The JSON prefix that precedes the label text in the response.
        Used for context-dependent tokenization."""
        return '{"label": "'

    # --- Logprob parsing ---

    @staticmethod
    def _parse_ollama_logprobs(response: Any) -> Optional[List[TokenLogprob]]:
        lps = getattr(response, "logprobs", None)
        if not lps:
            return None
        result = []
        for lp in lps:
            top = {}
            for alt in getattr(lp, "top_logprobs", []) or []:
                top[alt.token] = alt.logprob
            result.append(TokenLogprob(
                token=getattr(lp, "token", ""),
                logprob=getattr(lp, "logprob", 0.0),
                top_logprobs=top,
            ))
        return result

    # --- Sync interface ---

    def chat(
        self, messages, *, temperature=0.0, constrain_labels=None,
        logprobs=False, top_logprobs=5,
    ) -> ChatResponse:
        client = self._get_sync_client()
        fmt = self._build_json_enum(constrain_labels) if constrain_labels else None

        response = client.chat(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            format=fmt,
            logprobs=logprobs,
            top_logprobs=top_logprobs if logprobs else None,
            options={"temperature": temperature, "num_predict": self._max_tokens, **self._extra_body},
        )

        content = response.message.content
        label = ""
        try:
            label = json.loads(content).get("label", content)
        except (json.JSONDecodeError, TypeError):
            label = content

        return ChatResponse(
            content=content,
            label=label,
            logprobs=self._parse_ollama_logprobs(response),
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def score(self, messages: List[ChatMessage], completion: str) -> ScoringResponse:
        """Score a completion using Ollama's generate endpoint with suffix."""
        client = self._get_sync_client()
        prompt = "\n\n".join(m.content for m in messages if m.role in ("system", "user"))

        response = client.generate(
            model=self._model,
            prompt=prompt,
            suffix=completion,
            logprobs=True,
            options={"temperature": 0.0, "num_predict": 0, **self._extra_body},
        )

        return ScoringResponse(
            completion=completion,
            logprobs=self._parse_ollama_logprobs(response) or [],
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def tokenize(self, text: str, *, context: Optional[str] = None) -> List[Token]:
        """Tokenize text using Ollama's tokenize API.

        If context is provided, tokenizes context+text and returns only the
        tokens for text. This ensures context-dependent tokenization matches
        the actual response tokens (critical for JSON-wrapped labels).
        """
        client = self._get_sync_client()
        full_text = (context or "") + text
        response = client.tokenize(model=self._model, text=full_text)
        tokens = response.get("tokens", [])

        if context:
            ctx_response = client.tokenize(model=self._model, text=context)
            tokens = tokens[len(ctx_response.get("tokens", [])):]

        return [Token(text="", id=t) for t in tokens]

    # --- Async interface ---

    async def achat(self, messages, **kwargs) -> ChatResponse:
        client = await self._get_async_client()
        fmt = self._build_json_enum(kwargs.get("constrain_labels")) if kwargs.get("constrain_labels") else None

        response = await client.chat(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            format=fmt,
            logprobs=kwargs.get("logprobs", False),
            top_logprobs=kwargs.get("top_logprobs") if kwargs.get("logprobs") else None,
            options={"temperature": kwargs.get("temperature", 0.0), "num_predict": self._max_tokens, **self._extra_body},
        )

        content = response.message.content
        label = ""
        try:
            label = json.loads(content).get("label", content)
        except (json.JSONDecodeError, TypeError):
            label = content

        return ChatResponse(
            content=content, label=label,
            logprobs=self._parse_ollama_logprobs(response),
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    async def ascore(self, messages, completion: str) -> ScoringResponse:
        client = await self._get_async_client()
        prompt = "\n\n".join(m.content for m in messages if m.role in ("system", "user"))

        response = await client.generate(
            model=self._model, prompt=prompt, suffix=completion,
            logprobs=True,
            options={"temperature": 0.0, "num_predict": 0, **self._extra_body},
        )
        return ScoringResponse(
            completion=completion,
            logprobs=self._parse_ollama_logprobs(response) or [],
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    async def atokenize(self, text, *, context=None) -> List[Token]:
        client = await self._get_async_client()
        full_text = (context or "") + text
        response = await client.tokenize(model=self._model, text=full_text)
        tokens = response.get("tokens", [])
        if context:
            ctx_response = await client.tokenize(model=self._model, text=context)
            tokens = tokens[len(ctx_response.get("tokens", [])):]
        return [Token(text="", id=t) for t in tokens]
```

**Commit:**

```bash
git add src/ollama_classifier/backends/ollama.py
git commit -m "feat: add OllamaBackend with JSON enum constraint and context-dependent tokenization"
```

---

### Task 7: Update vLLM, SGLang, and llama.cpp backends

**Objective:** Update all three OpenAI-compatible backends to implement `constrain_labels`, `score()`, and `tokenize()` using their native constraint mechanisms.

**Files:**
- Modify: `src/ollama_classifier/backends/vllm.py`
- Modify: `src/ollama_classifier/backends/sglang.py`
- Modify: `src/ollama_classifier/backends/llamacpp.py`

**Key implementation per backend:**

#### vLLM

```python
class VLLMBackend(LLMBackend):
    @property
    def supports_bare_label_constraint(self) -> bool:
        return True  # guided_choice generates bare labels

    def _apply_constraint(self, body, labels):
        body["guided_choice"] = labels

    def score(self, messages, completion):
        # Use /v1/completions with prompt_logprobs to score the completion
        prompt = self._render_prompt(messages)
        # POST to /v1/completions with prompt=prompt+completion, echo=True, max_tokens=1
        # Extract logprobs only for completion tokens
        ...

    def tokenize(self, text, *, context=None):
        # POST to /v1/tokenize (vLLM extension)
        ...
```

#### SGLang

```python
class SGLangBackend(LLMBackend):
    @property
    def supports_bare_label_constraint(self) -> bool:
        return True  # regex generates bare labels

    def _apply_constraint(self, body, labels):
        # Build regex: (label1|label2|label3)
        escaped = [re.escape(l) for l in labels]
        body["regex"] = f"({'|'.join(escaped)})"
```

#### llama.cpp

```python
class LlamaCppBackend(LLMBackend):
    @property
    def supports_bare_label_constraint(self) -> bool:
        return True  # GBNF generates bare labels

    def _apply_constraint(self, body, labels):
        # Build GBNF grammar: root ::= "label1" | "label2" | "label3"
        quoted = [f'"{l}"' for l in labels]
        body["grammar"] = f"root ::= {' | '.join(quoted)}"
```

**Implementation notes for `score()` and `tokenize()` per backend:**

- **vLLM**: `score()` uses `/v1/completions` with `prompt_logprobs=0` and `echo=True`. `tokenize()` uses the `/tokenize` endpoint (vLLM extension, not standardized).
- **SGLang**: `score()` uses `/generate` with `return_logprob=True`. `tokenize()` uses internal endpoint.
- **llama.cpp**: `score()` uses `/completion` with `logprobs` and `suffix`. `tokenize()` uses `/tokenize`.

**Commit (per backend):**

```bash
git add src/ollama_classifier/backends/vllm.py
git commit -m "refactor: VLLMBackend with guided_choice, score, tokenize"

git add src/ollama_classifier/backends/sglang.py
git commit -m "refactor: SGLangBackend with regex constraint, score, tokenize"

git add src/ollama_classifier/backends/llamacpp.py
git commit -m "refactor: LlamaCppBackend with GBNF grammar, score, tokenize"
```

---

### Task 8: Implement unified LLMClassifier with adaptive generate()

**Objective:** Replace both `OllamaBackend` and `LLMClassifier` with a single class. The `generate()` method implements the full adaptive algorithm with `max_calls` budget and recursive cluster resolution.

**Files:**
- Rewrite: `src/ollama_classifier/classifier.py`
- Delete: `src/ollama_classifier/llm_classifier.py`

```python
# src/ollama_classifier/classifier.py
"""Unified LLM classifier with adaptive scoring.

Two methods:
    generate(): Adaptive constrained generation with divergence-aware
        confidence. Budget-controlled via max_calls. Ranges from 1 call
        (approximate) to ≤N calls (exact).
    classify(): Multi-call completion scoring with geometric-mean
        normalization. Always exact. N calls for N labels.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from .backends.base import ChatMessage, LLMBackend, TokenLogprob
from .prompts import build_classification_prompt, get_choice_labels
from .scoring import (
    Cluster, LabelTrie, divergence_point, geometric_mean_logprob,
    get_scored_lengths, identify_unresolved_clusters,
    score_labels_from_winning_path, stable_softmax,
)
from .types import ClassificationResult, ChoicesType


class LLMClassifier:
    """Backend-agnostic text classifier with two confidence scoring methods."""

    def __init__(self, backend: LLMBackend, *, max_workers: int = 4):
        self._backend = backend
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # ==================================================================
    # generate() — Adaptive trie-masked generation (Decision 3)
    # ==================================================================

    def generate(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
        *,
        max_calls: int | None = 1,
    ) -> ClassificationResult:
        """Adaptive constrained classification with divergence-aware confidence.

        Makes 1 to max_calls constrained API calls. Each call walks a prefix
        trie of label tokens. After each call, labels are scored up to their
        divergence point from the winning path. Unresolved clusters trigger
        supplementary calls.

        With max_calls=1: single call, partial scoring (fast, approximate).
        With max_calls=None: resolves everything recursively (exact).

        Args:
            text: Text to classify.
            choices: Labels as list or {label: description} dict.
            system_prompt: Optional custom system prompt.
            max_calls: Maximum number of API calls. None = unlimited.

        Returns:
            ClassificationResult with method="adaptive_generate".
        """
        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]

        # 1. Tokenize labels in the backend's constraint context
        token_context = self._get_token_context()
        token_sequences = self._tokenize_labels(labels, user, token_context)

        # 2. Build trie and determine required top_logprobs
        trie = LabelTrie()
        for label, tokens in token_sequences.items():
            trie.insert(label, tokens)
        k = max(trie.max_branching_factor, 5)

        # 3. Adaptive resolution loop
        all_step_logprobs: dict[str, list[float]] = {
            label: [] for label in labels
        }
        all_scored_lengths: dict[str, int] = {label: 0 for label in labels}
        calls_made = 0

        # First call: all labels
        frontier: list[Cluster] = [Cluster(labels=list(labels), resolved_length=0)]

        while frontier and (max_calls is None or calls_made < max_calls):
            cluster = frontier.pop(0)  # BFS for breadth-first resolution
            cluster_labels = cluster.labels
            resolved_len = cluster.resolved_length

            # Constrained call over this cluster
            response = self._backend.chat(
                messages=messages,
                temperature=0.0,
                constrain_labels=cluster_labels,
                logprobs=True,
                top_logprobs=k,
            )
            calls_made += 1

            # Extract per-step logprobs (filtering structural tokens for Ollama)
            step_lps = self._extract_step_logprobs(
                response, token_sequences, cluster_labels, resolved_len
            )

            # Score labels in this cluster up to divergence point
            winning_label = response.label
            cluster_token_seqs = {l: token_sequences[l] for l in cluster_labels}

            # If this is a sub-cluster, the winning path from the parent already
            # scored some tokens. We extend the scoring from resolved_len.
            cluster_scores = score_labels_from_winning_path(
                cluster_token_seqs, winning_label, step_lps
            )
            cluster_lengths = get_scored_lengths(cluster_token_seqs, winning_label)

            for label in cluster_labels:
                # Accumulate tokens scored in this call
                # For positions 0..resolved_len-1, tokens were already scored
                # by a parent call. For positions resolved_len..cluster_lengths[label]-1,
                # they are newly scored.
                new_len = cluster_lengths[label]
                if new_len > resolved_len:
                    # Extract the newly scored token logprobs
                    new_lps = []
                    for i in range(resolved_len, new_len):
                        token = token_sequences[label][i]
                        if i < len(step_lps) and token in step_lps[i]:
                            new_lps.append(step_lps[i][token])
                        else:
                            new_lps.append(float("-inf"))
                    all_step_logprobs[label].extend(new_lps)
                    all_scored_lengths[label] = new_len

            # Identify unresolved sub-clusters within this cluster
            sub_clusters = identify_unresolved_clusters(
                cluster_token_seqs, cluster_lengths
            )
            frontier.extend(sub_clusters)

        # 4. Compute final scores from accumulated logprobs
        raw_scores: dict[str, float] = {}
        coverage: dict[str, float] = {}
        for label in labels:
            lps = all_step_logprobs[label]
            total_tokens = len(token_sequences[label])
            if lps:
                raw_scores[label] = geometric_mean_logprob(lps)
            else:
                raw_scores[label] = float("-inf")
            coverage[label] = len(lps) / total_tokens if total_tokens > 0 else 1.0

        # 5. Softmax
        probabilities = stable_softmax(raw_scores)
        prediction = max(probabilities, key=probabilities.get)

        # 6. Determine approximation flag
        is_approximate = any(c < 1.0 for c in coverage.values())

        return ClassificationResult(
            prediction=prediction,
            confidence=probabilities[prediction],
            probabilities=probabilities,
            method="adaptive_generate",
            approximate=is_approximate,
            coverage=coverage,
            n_calls=calls_made,
            raw_response={
                "logprobs": raw_scores,
                "token_sequences": token_sequences,
                "step_logprobs": all_step_logprobs,
                "scored_lengths": all_scored_lengths,
            },
        )

    def _get_token_context(self) -> str | None:
        """Get the tokenization context for this backend.

        For backends that generate bare labels (vLLM, SGLang, llama.cpp),
        context is None — labels are tokenized standalone.

        For Ollama (JSON enum wrapper), context is the JSON prefix that
        precedes the label in the response: '{"label": "'.
        """
        if self._backend.supports_bare_label_constraint:
            return None
        else:
            return '{"label": "'

    def _tokenize_labels(
        self,
        labels: list[str],
        user_prompt: str,
        token_context: str | None,
    ) -> dict[str, list[str]]:
        """Tokenize each label in the appropriate context.

        For bare-label backends, tokenizes standalone label text.
        For Ollama, tokenizes label within the JSON prefix context.
        """
        token_sequences: dict[str, list[str]] = {}
        for label in labels:
            tokens = self._backend.tokenize(label, context=token_context)
            token_sequences[label] = [
                t.text if t.text else f"token_{t.id}" for t in tokens
            ]
        return token_sequences

    def _extract_step_logprobs(
        self,
        response,
        token_sequences: dict[str, list[str]],
        cluster_labels: list[str],
        offset: int,
    ) -> list[dict[str, float]]:
        """Extract per-step top_logprobs from a constrained call response.

        For bare-label backends, the response contains only label tokens.
        For Ollama (JSON wrapper), structural tokens are filtered by matching
        against known label tokens.
        """
        if not response.logprobs:
            return []

        # Collect all valid label tokens for filtering
        valid_tokens = set()
        for label in cluster_labels:
            valid_tokens.update(token_sequences[label])

        step_lps: list[dict[str, float]] = []
        for tlp in response.logprobs:
            # Filter top_logprobs to only include valid label tokens
            filtered = {
                tok: lp for tok, lp in tlp.top_logprobs.items()
                if tok in valid_tokens
            }
            if filtered:
                step_lps.append(filtered)
        return step_lps

    # ==================================================================
    # classify() — Multi-call completion scoring (Decision 2)
    # ==================================================================

    def classify(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Multi-call classification with geometric-mean completion scoring.

        For each label, scores the label as a completion of the prompt and
        extracts per-token logprobs WITHOUT generation. Applies geometric-mean
        normalization to eliminate token-count bias.

        Makes N API calls for N choices (parallelizable via async).

        Args:
            text: Text to classify.
            choices: Labels as list or {label: description} dict.
            system_prompt: Optional custom system prompt.

        Returns:
            ClassificationResult with method="multi_call", approximate=False.
        """
        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]

        raw_scores: dict[str, float] = {}
        logprob_details: dict[str, list[float]] = {}

        for label in labels:
            scoring = self._backend.score(messages, label)
            token_lps = [tlp.logprob for tlp in scoring.logprobs]
            if token_lps:
                raw_scores[label] = geometric_mean_logprob(token_lps)
            else:
                raw_scores[label] = float("-inf")
            logprob_details[label] = token_lps

        probabilities = stable_softmax(raw_scores)
        prediction = max(probabilities, key=probabilities.get)

        return ClassificationResult(
            prediction=prediction,
            confidence=probabilities[prediction],
            probabilities=probabilities,
            method="multi_call",
            approximate=False,
            n_calls=len(labels),
            raw_response={"logprobs": raw_scores, "token_logprobs": logprob_details},
        )

    # ==================================================================
    # Batch methods (parallelized)
    # ==================================================================

    def batch_generate(
        self, texts, choices, system_prompt=None, *, max_calls=None,
    ) -> list[ClassificationResult]:
        return list(self._executor.map(
            lambda t: self.generate(t, choices, system_prompt, max_calls=max_calls), texts
        ))

    def batch_classify(
        self, texts, choices, system_prompt=None,
    ) -> list[ClassificationResult]:
        return list(self._executor.map(
            lambda t: self.classify(t, choices, system_prompt), texts
        ))

    # ==================================================================
    # Async methods
    # ==================================================================

    async def agenerate(self, text, choices, system_prompt=None, *, max_calls=None):
        """Async adaptive generation."""
        # Mirrors generate() but with await for each backend call.
        # Cluster calls can be parallelized with asyncio.gather when frontier
        # contains multiple independent clusters.
        ...  # Full implementation mirrors generate() with async backend calls

    async def aclassify(self, text, choices, system_prompt=None):
        """Async multi-call classification (labels scored concurrently)."""
        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]

        score_tasks = [self._backend.ascore(messages, label) for label in labels]
        scoring_results = await asyncio.gather(*score_tasks)

        raw_scores: dict[str, float] = {}
        for label, scoring in zip(labels, scoring_results):
            token_lps = [tlp.logprob for tlp in scoring.logprobs]
            raw_scores[label] = (
                geometric_mean_logprob(token_lps) if token_lps else float("-inf")
            )

        probabilities = stable_softmax(raw_scores)
        prediction = max(probabilities, key=probabilities.get)

        return ClassificationResult(
            prediction=prediction,
            confidence=probabilities[prediction],
            probabilities=probabilities,
            method="multi_call",
            approximate=False,
            n_calls=len(labels),
            raw_response={"logprobs": raw_scores},
        )

    async def abatch_generate(self, texts, choices, system_prompt=None, *, max_calls=None):
        return await asyncio.gather(
            *[self.agenerate(t, choices, system_prompt, max_calls=max_calls) for t in texts]
        )

    async def abatch_classify(self, texts, choices, system_prompt=None):
        return await asyncio.gather(
            *[self.aclassify(t, choices, system_prompt) for t in texts]
        )
```

**Commit:**

```bash
git rm src/ollama_classifier/llm_classifier.py
git add src/ollama_classifier/classifier.py
git commit -m "feat: unified LLMClassifier with adaptive generate() and multi-call classify()"
```

---

### Task 9: Update `__init__.py` and backends `__init__.py`

**Objective:** Export the new unified API.

**Files:**
- Modify: `src/ollama_classifier/__init__.py`
- Modify: `src/ollama_classifier/backends/__init__.py`

```python
# src/ollama_classifier/__init__.py
from .types import ClassificationResult, ChoicesType
from .classifier import LLMClassifier

__all__ = ["LLMClassifier", "ClassificationResult", "ChoicesType"]
__version__ = "0.4.0"
```

```python
# src/ollama_classifier/backends/__init__.py
from .base import (
    LLMBackend, ChatMessage, ChatResponse, ScoringResponse, TokenLogprob, Token,
)
from .ollama import OllamaBackend
from .vllm import VLLMBackend
from .sglang import SGLangBackend
from .llamacpp import LlamaCppBackend

__all__ = [
    "LLMBackend", "ChatMessage", "ChatResponse", "ScoringResponse",
    "TokenLogprob", "Token",
    "OllamaBackend", "VLLMBackend", "SGLangBackend", "LlamaCppBackend",
]
```

**Commit:**

```bash
git add src/ollama_classifier/__init__.py src/ollama_classifier/backends/__init__.py
git commit -m "refactor: update exports for unified architecture"
```

---

### Task 10: Update `pyproject.toml`

**Objective:** Make `httpx` and `pydantic` required, `ollama` optional (≥0.12), add `pytest`, bump version.

```toml
[project]
name = "ollama-classifier"
version = "0.4.0"
description = "Text classification with constrained output and confidence scoring across multiple LLM backends"
readme = "README.md"
authors = [
    {name = "Luigi Palumbo"},
    {name = "Mengting Yu"},
    {name = "Carolina Camassa"},
]
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
ollama = [
    "ollama>=0.12.0",
]
docs = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=2.0.0",
    "myst-parser>=2.0.0",
]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]

[build-system]
requires = ["uv_build>=0.9.26,<0.10.0"]
build-backend = "uv_build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

```bash
uv lock
git add pyproject.toml uv.lock
git commit -m "build: make httpx/pydantic required, ollama>=0.12 optional, add pytest, bump 0.4.0"
```

---

### Task 11: Update examples and documentation

**Objective:** Update all examples, docs, README, and changelog.

**Files:**
- Modify: `examples/example_usage.py`
- Modify: `examples/run_sample_data.py`
- Modify: `README.md`
- Modify: `docs/usage.rst`
- Modify: `docs/api.rst`
- Modify: `docs/backends.rst`
- Modify: `docs/changelog.rst`

**Key documentation changes:**

1. **All examples** migrate from `OllamaClassifier(Client(), model=...)` to `LLMClassifier(OllamaBackend(model=...))`.

2. **New "Choosing a Scoring Method" section:**

```rst
Choosing a Scoring Method
-------------------------

+-------------------+-------------------+---------------+-----------------------------+
| Method            | API Calls         | Exactness     | When to Use                 |
+===================+===================+===============+=============================+
| generate()        | 1 to max_calls    | Adaptive      | Speed-critical. With         |
| max_calls=1       |                   | (approximate) | max_calls=None, becomes exact|
+-------------------+-------------------+---------------+-----------------------------+
| classify()        | N (one per label) | Always exact  | Research, calibration       |
+-------------------+-------------------+---------------+-----------------------------+
```

3. **Add `ClassificationResult.method`, `.approximate`, `.coverage`, `.n_calls`** to the working-with-results docs.

4. **Changelog entry for 0.4.0** (see Task 13 for full text).

**Commit:**

```bash
git add examples/ docs/ README.md
git commit -m "docs: update all examples and documentation for v0.4.0"
```

---

### Task 12: Integration tests with mocked backends

**Objective:** End-to-end tests for both scoring methods using a mock backend.

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_classifier.py`

```python
# tests/conftest.py
import pytest
from ollama_classifier.backends.base import (
    ChatMessage, ChatResponse, LLMBackend, ScoringResponse, Token, TokenLogprob,
)


class MockBackend(LLMBackend):
    """Mock backend with configurable responses for testing."""

    def __init__(
        self,
        label_tokens: dict[str, list[str]] | None = None,
        step_logprobs_map: dict[str, list[dict[str, float]]] | None = None,
        completion_logprobs: dict[str, list[float]] | None = None,
    ):
        super().__init__(model="mock", base_url="http://mock")
        self._label_tokens = label_tokens or {}
        self._step_logprobs_map = step_logprobs_map or {}
        self._completion_logprobs = completion_logprobs or {}
        self.call_count = 0

    @property
    def supports_bare_label_constraint(self) -> bool:
        return True

    def chat(self, messages, *, temperature=0.0, constrain_labels=None,
             logprobs=False, top_logprobs=5) -> ChatResponse:
        self.call_count += 1
        labels = constrain_labels or list(self._label_tokens.keys())
        # Return the first label as winner (deterministic for testing)
        winner = labels[0]
        step_lps = self._step_logprobs_map.get(winner, [])

        lp_objects = []
        for step_lp in step_lps:
            best_token = max(step_lp, key=step_lp.get)
            lp_objects.append(TokenLogprob(
                token=best_token, logprob=step_lp[best_token],
                top_logprobs=dict(step_lp),
            ))

        return ChatResponse(content=winner, label=winner, logprobs=lp_objects)

    def score(self, messages, completion: str) -> ScoringResponse:
        self.call_count += 1
        lps = self._completion_logprobs.get(completion, [-1.0])
        return ScoringResponse(
            completion=completion,
            logprobs=[TokenLogprob(token="x", logprob=lp) for lp in lps],
        )

    def tokenize(self, text, *, context=None) -> list[Token]:
        return [Token(text=t, id=i) for i, t in enumerate(
            self._label_tokens.get(text, [text])
        )]

    async def achat(self, *args, **kwargs):
        return self.chat(*args, **kwargs)

    async def ascore(self, *args, **kwargs):
        return self.score(*args, **kwargs)

    async def atokenize(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)


@pytest.fixture
def mock_backend_single_token():
    """Labels that are single tokens — generate() is exact."""
    return MockBackend(
        label_tokens={"positive": ["positive"], "negative": ["negative"], "neutral": ["neutral"]},
        step_logprobs_map={
            "positive": [{"positive": -0.3, "negative": -1.5, "neutral": -2.8}],
        },
        completion_logprobs={
            "positive": [-0.3], "negative": [-1.5], "neutral": [-2.8],
        },
    )


@pytest.fixture
def mock_backend_multi_token():
    """Labels with shared prefixes — tests divergence-aware scoring."""
    return MockBackend(
        label_tokens={
            "a": ["t1", "t2", "t3", "t4a"],
            "b": ["t1", "t2", "t3", "t4b"],
            "c": ["t1c", "t2c", "t3c", "t4c"],
        },
        step_logprobs_map={
            # Winner a: b diverges at pos 3, c at pos 0
            "a": [
                {"t1": -0.1, "t1c": -2.5},
                {"t2": -0.2, "x": -3.0},
                {"t3": -0.15, "y": -2.8},
                {"t4a": -0.1, "t4b": -1.5},
            ],
            # Winner b: a diverges at pos 3
            "b": [
                {"t1": -0.1, "t1c": -2.5},
                {"t2": -0.2, "x": -3.0},
                {"t3": -0.15, "y": -2.8},
                {"t4b": -0.1, "t4a": -1.0},
            ],
        },
    )
```

```python
# tests/test_classifier.py
import pytest
from ollama_classifier import LLMClassifier, ClassificationResult


class TestGenerate:
    def test_generate_returns_result(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        result = clf.generate("test", ["positive", "negative", "neutral"])
        assert isinstance(result, ClassificationResult)
        assert result.prediction == "positive"
        assert result.method == "adaptive_generate"
        assert result.n_calls == 1

    def test_generate_probabilities_sum_to_one(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        result = clf.generate("test", ["positive", "negative", "neutral"])
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-10

    def test_generate_single_token_exact(self, mock_backend_single_token):
        """Single-token labels → fully exact, not approximate."""
        clf = LLMClassifier(mock_backend_single_token)
        result = clf.generate("test", ["positive", "negative", "neutral"])
        assert result.approximate is False
        assert all(c == 1.0 for c in result.coverage.values())

    def test_generate_multi_token_partial(self, mock_backend_multi_token):
        """max_calls=1 with multi-token labels → partial coverage."""
        clf = LLMClassifier(mock_backend_multi_token)
        result = clf.generate("test", ["a", "b", "c"], max_calls=1)
        assert result.approximate is True
        assert result.coverage["a"] == 1.0   # winner fully resolved
        assert result.coverage["b"] == 1.0   # b diverges at pos 3, all 4 tokens scored
        assert result.coverage["c"] == 0.25  # c diverges at pos 0, only 1/4 tokens

    def test_generate_multi_token_exact_with_max_calls_none(self, mock_backend_multi_token):
        """max_calls=None resolves everything recursively."""
        clf = LLMClassifier(mock_backend_multi_token)
        result = clf.generate("test", ["a", "b", "c"], max_calls=None)
        assert result.approximate is False
        assert all(c == 1.0 for c in result.coverage.values())
        # Should take 2 calls: first resolves a (winner), then cluster {a,b} wait—
        # Actually a and b are both resolved in call 1. c is the only unresolved.
        # But c is alone in its cluster (no siblings), so it's resolved in call 2.
        assert result.n_calls == 2


class TestClassify:
    def test_classify_returns_result(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        result = clf.classify("test", ["positive", "negative", "neutral"])
        assert isinstance(result, ClassificationResult)
        assert result.prediction == "positive"
        assert result.method == "multi_call"
        assert result.approximate is False
        assert result.n_calls == 3

    def test_classify_confidence_not_concentrated(self, mock_backend_single_token):
        """Verify geometric-mean normalization prevents concentration."""
        clf = LLMClassifier(mock_backend_single_token)
        result = clf.classify("test", ["positive", "negative", "neutral"])
        assert result.confidence < 0.99  # was ~0.9999 with raw sum

    def test_classify_probabilities_sum_to_one(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        result = clf.classify("test", ["positive", "negative", "neutral"])
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-10


class TestBatch:
    def test_batch_generate(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        results = clf.batch_generate(["a", "b"], ["positive", "negative"])
        assert len(results) == 2
        assert all(r.method == "adaptive_generate" for r in results)

    def test_batch_classify(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        results = clf.batch_classify(["a", "b"], ["positive", "negative"])
        assert len(results) == 2
        assert all(r.method == "multi_call" for r in results)
```

**Run all tests:**

```bash
uv run pytest tests/ -v
```

**Commit:**

```bash
git add tests/
git commit -m "test: add integration tests for adaptive generate() and multi-call classify()"
```

---

### Task 13: Final cleanup, lint, and verification

**Objective:** Run linters, verify imports, clean up dead code, update changelog.

**Changelog entry:**

```rst
[0.4.0] - 2026-07-06
--------------------

Added
~~~~~

- ``OllamaBackend`` — wraps Ollama SDK (≥0.12) behind the ``LLMBackend`` interface
- Adaptive ``generate()`` method with divergence-aware confidence scoring and
  ``max_calls`` budget parameter
- ``coverage`` field on ``ClassificationResult`` showing per-label token coverage
- ``n_calls`` field on ``ClassificationResult``
- ``max_calls`` parameter on ``generate()``: ``1`` (fast, approximate), ``K``
  (adaptive), ``None`` (exact)
- Geometric-mean (length-normalized) confidence scoring, applied consistently
  across both methods
- ``scoring.py`` module with pure functions: ``geometric_mean_logprob``,
  ``stable_softmax``, ``LabelTrie``, divergence-aware path reconstruction
- Comprehensive test suite (unit tests for scoring + integration tests with
  mocked backends)
- Parallelized sync batch methods via ``ThreadPoolExecutor``

Changed
~~~~~~~

- **Breaking:** ``generate()`` now returns ``ClassificationResult`` (was ``str``).
  Access ``.prediction`` for the label string.
- **Breaking:** ``OllamaClassifier`` and ``LLMClassifier`` (old) replaced by
  unified ``LLMClassifier`` accepting any ``LLMBackend``
- ``classify()`` rewritten: uses completion scoring (``backend.score()``) with
  geometric-mean normalization instead of forced-generation logprob sums
- ``ClassificationResult`` migrated to Pydantic with new fields: ``method``,
  ``approximate``, ``coverage``, ``n_calls``
- ``httpx`` and ``pydantic`` are now required dependencies
- ``ollama`` SDK is now optional (``[ollama]`` extra, requires ≥0.12)
- All backends implement ``constrain_labels``, ``score``, ``tokenize``

Fixed
~~~~~

- Confidence concentration bias caused by summing raw logprobs over labels
  with different token lengths (replaced with geometric-mean normalization)
- Silent ``0.0`` fallback when logprobs missing (now returns ``-inf``)
- Duplicated classifier code (~400 LOC eliminated)

Removed
~~~~~~~

- ``OllamaClassifier`` class (replaced by ``LLMClassifier`` + ``OllamaBackend``)
- Old ``LLMClassifier`` from ``llm_classifier.py`` (merged into ``classifier.py``)
- ``[backends]`` optional dependency group (httpx is now required)
```

**Steps:**

1. `uv sync --all-extras`
2. `uv run ruff check src/ tests/ && uv run ruff format src/ tests/`
3. Verify imports: `uv run python -c "from ollama_classifier import LLMClassifier, ClassificationResult; print('OK')"`
4. `uv run pytest tests/ -v --tb=short`
5. Verify no references to old classes: `grep -r "OllamaClassifier" src/ examples/ docs/ --include="*.py" --include="*.rst"` (should only appear in changelog)

**Commit:**

```bash
git add -A
git commit -m "chore: lint, format, verify, update changelog for v0.4.0"
```

---

## Summary of Changes

| Area | Before (v0.3.0) | After (v0.4.0) |
|---|---|---|
| Classifier classes | 2 (`OllamaClassifier` + `LLMClassifier`, 90% duplicated) | 1 (`LLMClassifier`) |
| Backends | 3 + Ollama SDK inline | 4 unified (incl. `OllamaBackend`) |
| `generate()` | Returns `str` | Returns `ClassificationResult` with adaptive confidence |
| `generate()` confidence | None | Divergence-aware, `max_calls` budget, exact when resolved |
| `classify()` | N forced-gen calls, raw sum (biased) | N completion-scoring calls, geometric-mean (exact) |
| Confidence quality | Catastrophically over-concentrated | Calibrated (multi-call) or adaptive (generate) |
| Length bias | Present | Eliminated (geometric-mean normalization) |
| Tokenization | N/A | Context-dependent (Ollama JSON wrapper handled) |
| `httpx` | Optional | Required |
| `ollama` SDK | Required | Optional (≥0.12) |
| Tests | None | 30+ tests (unit + integration) |
| `ClassificationResult` | dataclass | Pydantic model with `method`, `approximate`, `coverage`, `n_calls` |
| Batch sync | Sequential loop | `ThreadPoolExecutor` |
