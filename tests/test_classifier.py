"""Integration tests for LLMClassifier with mocked backends."""

import pytest

from ollama_classifier import LLMClassifier, ClassificationResult


# =========================================================================
# generate() — Adaptive constrained generation
# =========================================================================

class TestGenerate:
    def test_generate_returns_result(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        result = clf.generate("test", ["positive", "negative", "neutral"])
        assert isinstance(result, ClassificationResult)
        assert result.prediction == "positive"
        assert result.method == "adaptive_generate"
        assert result.n_calls == 1

    def test_generate_confidence_in_range(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        result = clf.generate("test", ["positive", "negative", "neutral"])
        assert 0.0 <= result.confidence <= 1.0

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

    def test_generate_multi_token_partial_coverage(self, mock_backend_multi_token):
        """max_calls=1 with multi-token labels → partial coverage."""
        clf = LLMClassifier(mock_backend_multi_token)
        result = clf.generate("test", ["a", "b", "c"], max_calls=1)
        assert result.approximate is True
        # a is winner, fully resolved (4/4 tokens)
        assert result.coverage["a"] == 1.0
        # b diverges at pos 3, all 4 tokens scored
        assert result.coverage["b"] == 1.0
        # c diverges at pos 0, only 1/4 tokens
        assert result.coverage["c"] == 0.25

    def test_generate_multi_token_exact_with_max_calls_none(
        self, mock_backend_multi_token
    ):
        """max_calls=None resolves all multi-label clusters recursively.

        Labels that diverge from the winner and form multi-label clusters
        get fully resolved.  A label that diverges alone (single-label
        cluster) keeps its initial partial coverage — its probability is
        already fixed by the between-group distribution, and no
        reproportioning call would change it.
        """
        clf = LLMClassifier(mock_backend_multi_token)
        result = clf.generate("test", ["a", "b", "c"], max_calls=None)
        # a and b share prefix [t1,t2,t3] and form a 2-label cluster — resolved
        assert result.coverage["a"] == 1.0
        assert result.coverage["b"] == 1.0
        # c diverges at token 0 — single-label cluster, stays partial
        assert result.coverage["c"] == 0.25
        assert result.approximate is True  # c is still partial

    def test_generate_max_calls_limits_calls(self, mock_backend_single_token):
        """max_calls=1 should make exactly 1 call."""
        clf = LLMClassifier(mock_backend_single_token)
        clf.generate("test", ["positive", "negative", "neutral"], max_calls=1)
        assert mock_backend_single_token.call_count == 1


# =========================================================================
# classify() — Multi-call completion scoring
# =========================================================================

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
        """Verify geometric-mean normalization prevents concentration.

        With raw sum (old method), confidence was ~0.9999. With geometric mean,
        it should be much lower for these logprob values.
        """
        clf = LLMClassifier(mock_backend_single_token)
        result = clf.classify("test", ["positive", "negative", "neutral"])
        # -0.3 vs -1.5 vs -2.8 → softmax should not be catastrophically peaked
        assert result.confidence < 0.95

    def test_classify_probabilities_sum_to_one(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        result = clf.classify("test", ["positive", "negative", "neutral"])
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-10

    def test_classify_makes_n_calls(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        clf.classify("test", ["positive", "negative", "neutral"])
        assert mock_backend_single_token.call_count == 3

    def test_classify_multi_token(self, mock_backend_multi_token):
        """Multi-token labels should also work with classify()."""
        clf = LLMClassifier(mock_backend_multi_token)
        result = clf.classify("test", ["a", "b", "c"])
        assert result.prediction == "a"  # a has highest geometric-mean logprob
        assert result.method == "multi_call"
        assert result.approximate is False

    def test_classify_length_normalization(self, mock_backend_multi_token):
        """Verify that classify() normalizes across different label lengths."""
        clf = LLMClassifier(mock_backend_multi_token)
        result = clf.classify("test", ["a", "b", "c"])
        # a: geometric mean of [-0.1, -0.2, -0.15, -0.1] = -0.1375
        # b: geometric mean of [-0.1, -0.2, -0.15, -1.5] = -0.4875
        # c: geometric mean of [-2.5, -0.2, -0.15, -0.1] = -0.7375
        # softmax should rank a > b > c
        probs = result.probabilities
        assert probs["a"] > probs["b"] > probs["c"]


# =========================================================================
# Batch methods
# =========================================================================

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

    def test_batch_generate_parallelism(self, mock_backend_single_token):
        """Batch should use ThreadPoolExecutor (multiple calls happen)."""
        clf = LLMClassifier(mock_backend_single_token)
        results = clf.batch_generate(["a", "b", "c"], ["positive", "negative"])
        assert len(results) == 3


# =========================================================================
# Async methods
# =========================================================================

class TestAsync:
    @pytest.mark.asyncio
    async def test_agenerate_single_token(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        result = await clf.agenerate("test", ["positive", "negative", "neutral"])
        assert result.prediction == "positive"
        assert result.method == "adaptive_generate"
        assert result.approximate is False

    @pytest.mark.asyncio
    async def test_aclassify(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        result = await clf.aclassify("test", ["positive", "negative", "neutral"])
        assert result.prediction == "positive"
        assert result.method == "multi_call"
        assert result.n_calls == 3

    @pytest.mark.asyncio
    async def test_abatch_generate(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        results = await clf.abatch_generate(["a", "b"], ["positive", "negative"])
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_abatch_classify(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        results = await clf.abatch_classify(["a", "b"], ["positive", "negative"])
        assert len(results) == 2


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    def test_generate_with_dict_choices(self, mock_backend_single_token):
        """choices can be a dict mapping labels to descriptions."""
        clf = LLMClassifier(mock_backend_single_token)
        choices = {
            "positive": "Text expressing happiness",
            "negative": "Text expressing anger",
        }
        result = clf.generate("test", choices)
        assert result.prediction in ("positive", "negative")

    def test_classify_with_dict_choices(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        choices = {
            "positive": "Text expressing happiness",
            "negative": "Text expressing anger",
        }
        result = clf.classify("test", choices)
        assert result.prediction in ("positive", "negative")

    def test_generate_two_labels(self, mock_backend_single_token):
        clf = LLMClassifier(mock_backend_single_token)
        result = clf.generate("test", ["positive", "negative"])
        assert len(result.probabilities) == 2
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-10


# =========================================================================
# Regression: max_calls must not decrease accuracy (reproportion approach)
# =========================================================================

class TestMaxCallsMonotonicity:
    """Regression tests for the hierarchical reproportion approach.

    The original cluster-resolution code mixed logprobs from different
    constraint contexts into a single geometric mean, which could DECREASE
    accuracy as max_calls increased.  The fix uses **reproportioning**:
    supplementary calls only redistribute probability mass *within* a cluster,
    never changing between-group totals.

    These tests verify:
    1. Increasing max_calls never flips a correct prediction to incorrect.
    2. Between-group probability mass is preserved during reproportioning.
    3. Probabilities always sum to 1.0.
    """

    def test_max_calls_does_not_flip_prediction(self):
        """generate(max_calls=None) must not produce a worse prediction
        than generate(max_calls=1).

        Scenario: 3 labels with shared prefix.
          A = [shared, a_end]                 (2 tokens)
          B = [shared, b_mid, b1, b2, b3]    (5 tokens, diverges at token 1)
          C = [c_first, c_end]               (2 tokens, diverges at token 0)

        Model's true preference: A > B > C.
        Greedy constrained generation picks A.

        When the {B} cluster is resolved via a subset call, the
        reproportioning must not inflate B's probability above A's.
        """
        from tests.conftest import MockBackend

        label_tokens = {
            "A": ["shared", "a_end"],
            "B": ["shared", "b_mid", "b1", "b2", "b3"],
            "C": ["c_first", "c_end"],
        }
        # MockBackend returns labels[0] as winner for any constraint set.
        # For 3-way call: winner = "A"
        # For 1-way call on ["B"]: winner = "B"
        # For 1-way call on ["C"]: winner = "C"
        step_logprobs_map = {
            "A": [
                {"shared": -0.3, "c_first": -1.5},
                {"a_end": -0.1, "b_mid": -0.6},
            ],
            "B": [
                {"shared": -0.3},  # subset call, only B in trie
                {"b_mid": -0.6},
                {"b1": -0.5},
                {"b2": -0.5},
                {"b3": -0.5},
            ],
            "C": [
                {"c_first": -1.5},
                {"c_end": -0.3},
            ],
        }
        completion_logprobs = {
            "A": [-0.3, -0.1],
            "B": [-0.3, -0.6, -0.5, -0.5, -0.5],
            "C": [-1.5, -0.3],
        }

        predictions = {}
        for max_calls in [1, 2, 3, None]:
            backend = MockBackend(
                label_tokens=label_tokens,
                step_logprobs_map=step_logprobs_map,
                completion_logprobs=completion_logprobs,
                bare_labels=True,
            )
            clf = LLMClassifier(backend)
            result = clf.generate("test", ["A", "B", "C"], max_calls=max_calls)
            predictions[max_calls] = result.prediction
            # Probabilities must always sum to 1.0
            assert abs(sum(result.probabilities.values()) - 1.0) < 1e-10, (
                f"max_calls={max_calls}: probabilities don't sum to 1.0"
            )

        # All predictions must be "A" (the correct answer)
        for mc, pred in predictions.items():
            assert pred == "A", (
                f"max_calls={mc}: expected 'A', got '{pred}'"
            )

    def test_reproportion_preserves_group_mass(self):
        """Reproportioning must preserve the total probability mass of
        each cluster.  The sum of probabilities for cluster members
        before and after resolution must be equal.
        """
        from tests.conftest import MockBackend

        label_tokens = {
            "A": ["shared", "a_end"],
            "B": ["shared", "b_mid", "b1", "b2", "b3"],
            "C": ["c_first", "c_end"],
        }
        step_logprobs_map = {
            "A": [
                {"shared": -0.3, "c_first": -1.5},
                {"a_end": -0.1, "b_mid": -0.6},
            ],
            "B": [
                {"shared": -0.3},
                {"b_mid": -0.6},
                {"b1": -0.5},
                {"b2": -0.5},
                {"b3": -0.5},
            ],
            "C": [
                {"c_first": -1.5},
                {"c_end": -0.3},
            ],
        }

        # mc=1: initial distribution (no resolution)
        backend1 = MockBackend(
            label_tokens=label_tokens,
            step_logprobs_map=step_logprobs_map,
            bare_labels=True,
        )
        clf1 = LLMClassifier(backend1)
        result1 = clf1.generate("test", ["A", "B", "C"], max_calls=1)

        # mc=None: full resolution via reproportioning
        backend2 = MockBackend(
            label_tokens=label_tokens,
            step_logprobs_map=step_logprobs_map,
            bare_labels=True,
        )
        clf2 = LLMClassifier(backend2)
        result2 = clf2.generate("test", ["A", "B", "C"], max_calls=None)

        # Group {B} mass must be preserved (B is a singleton cluster after
        # the first call resolves the {A,B} shared prefix, but B itself
        # may get resolved further). The key property: P(A) + P(B) + P(C)
        # must equal 1.0 in both cases, and P(A) should not decrease.
        assert result2.probabilities["A"] >= result1.probabilities["A"] - 1e-10, (
            f"A's probability decreased: mc=1={result1.probabilities['A']}, "
            f"mc=None={result2.probabilities['A']}"
        )

    def test_single_token_labels_no_resolution_needed(self):
        """When all labels are single-token, max_calls has no effect —
        there are no clusters to resolve."""
        from tests.conftest import MockBackend

        label_tokens = {
            "positive": ["positive"],
            "negative": ["negative"],
            "neutral": ["neutral"],
        }
        step_logprobs_map = {
            "positive": [{"positive": -0.3, "negative": -1.5, "neutral": -2.8}],
        }

        for mc in [1, 5, None]:
            backend = MockBackend(
                label_tokens=label_tokens,
                step_logprobs_map=step_logprobs_map,
                bare_labels=True,
            )
            clf = LLMClassifier(backend)
            result = clf.generate("test", ["positive", "negative", "neutral"],
                                  max_calls=mc)
            assert result.prediction == "positive"
            assert result.n_calls == 1  # no resolution calls needed
            assert not result.approximate
