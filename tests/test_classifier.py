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
        """max_calls=None resolves everything recursively."""
        clf = LLMClassifier(mock_backend_multi_token)
        result = clf.generate("test", ["a", "b", "c"], max_calls=None)
        assert result.approximate is False
        assert all(c == 1.0 for c in result.coverage.values())

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
