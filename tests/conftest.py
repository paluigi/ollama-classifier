"""Test fixtures: MockBackend and shared fixtures."""

import pytest
from ollama_classifier.backends.base import (
    ChatMessage,
    ChatResponse,
    LLMBackend,
    ScoringResponse,
    Token,
    TokenLogprob,
)


class MockBackend(LLMBackend):
    """Mock backend with configurable responses for testing.

    Simulates constrained generation and completion scoring without a real
    LLM server. Returns deterministic responses based on pre-configured
    label token sequences and step logprobs.
    """

    def __init__(
        self,
        label_tokens: dict[str, list[str]] | None = None,
        step_logprobs_map: dict[str, list[dict[str, float]]] | None = None,
        completion_logprobs: dict[str, list[float]] | None = None,
        bare_labels: bool = True,
    ):
        super().__init__(model="mock", base_url="http://mock")
        self._label_tokens = label_tokens or {}
        self._step_logprobs_map = step_logprobs_map or {}
        self._completion_logprobs = completion_logprobs or {}
        self._bare_labels = bare_labels
        self.call_count = 0

    @property
    def supports_bare_label_constraint(self) -> bool:
        return self._bare_labels

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        constrain_labels: list[str] | None = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> ChatResponse:
        self.call_count += 1
        labels = constrain_labels or list(self._label_tokens.keys())
        # Return the first label as winner (deterministic for testing)
        winner = labels[0]
        step_lps = self._step_logprobs_map.get(winner, [])

        lp_objects: list[TokenLogprob] = []
        for step_lp in step_lps:
            best_token = max(step_lp, key=step_lp.get)
            lp_objects.append(
                TokenLogprob(
                    token=best_token,
                    logprob=step_lp[best_token],
                    top_logprobs=dict(step_lp),
                )
            )

        return ChatResponse(content=winner, label=winner, logprobs=lp_objects)

    def score(self, messages: list[ChatMessage], completion: str) -> ScoringResponse:
        self.call_count += 1
        lps = self._completion_logprobs.get(completion, [-1.0])
        return ScoringResponse(
            completion=completion,
            logprobs=[TokenLogprob(token="x", logprob=lp) for lp in lps],
        )

    def tokenize(self, text: str, *, context: str | None = None) -> list[Token]:
        return [
            Token(text=t, id=i)
            for i, t in enumerate(self._label_tokens.get(text, [text]))
        ]

    async def achat(self, *args, **kwargs) -> ChatResponse:
        return self.chat(*args, **kwargs)

    async def ascore(self, *args, **kwargs) -> ScoringResponse:
        return self.score(*args, **kwargs)

    async def atokenize(self, *args, **kwargs) -> list[Token]:
        return self.tokenize(*args, **kwargs)


@pytest.fixture
def mock_backend_single_token():
    """Labels that are single tokens — generate() is exact."""
    return MockBackend(
        label_tokens={
            "positive": ["positive"],
            "negative": ["negative"],
            "neutral": ["neutral"],
        },
        step_logprobs_map={
            "positive": [{"positive": -0.3, "negative": -1.5, "neutral": -2.8}],
        },
        completion_logprobs={
            "positive": [-0.3],
            "negative": [-1.5],
            "neutral": [-2.8],
        },
    )


@pytest.fixture
def mock_backend_multi_token():
    """Labels with shared prefixes — tests divergence-aware scoring.

    Labels a and b share prefix [t1, t2, t3] and diverge at token 3.
    Label c diverges from both at token 0.
    """
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
        completion_logprobs={
            "a": [-0.1, -0.2, -0.15, -0.1],
            "b": [-0.1, -0.2, -0.15, -1.5],
            "c": [-2.5, -0.2, -0.15, -0.1],
        },
    )
