"""Tests for the scoring module: probability functions, trie, divergence-aware scoring."""

import math
import pytest

from ollama_classifier.scoring import (
    geometric_mean_logprob,
    stable_softmax,
    LabelTrie,
    TrieNode,
    divergence_point,
    score_labels_from_winning_path,
    get_scored_lengths,
    identify_unresolved_clusters,
    Cluster,
)


# =========================================================================
# geometric_mean_logprob
# =========================================================================

class TestGeometricMeanLogprob:
    def test_single_token(self):
        assert geometric_mean_logprob([-0.5]) == -0.5

    def test_multi_token(self):
        lps = [math.log(0.9)] * 4
        expected = sum(lps) / 4
        assert abs(geometric_mean_logprob(lps) - expected) < 1e-10

    def test_removes_length_bias(self):
        """Labels with identical per-token confidence get identical scores."""
        short = [math.log(0.95)]
        long_ = [math.log(0.95)] * 4
        assert abs(geometric_mean_logprob(short) - geometric_mean_logprob(long_)) < 1e-10

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            geometric_mean_logprob([])

    def test_all_inf_returns_neg_inf(self):
        assert geometric_mean_logprob([float("-inf"), float("-inf")]) == float("-inf")

    def test_mix_of_valid_and_inf(self):
        lps = [-0.5, float("-inf"), -1.0]
        expected = (-0.5 + -1.0) / 2
        assert abs(geometric_mean_logprob(lps) - expected) < 1e-10


# =========================================================================
# stable_softmax
# =========================================================================

class TestStableSoftmax:
    def test_basic(self):
        probs = stable_softmax({"a": -0.1, "b": -2.0, "c": -5.0})
        assert abs(sum(probs.values()) - 1.0) < 1e-10
        assert probs["a"] > probs["b"] > probs["c"]

    def test_stability_with_extreme_values(self):
        probs = stable_softmax({"short": -0.02, "long": -15.0})
        assert probs["short"] > probs["long"]
        assert probs["short"] < 1.0

    def test_handles_inf(self):
        probs = stable_softmax({"a": -1.0, "b": float("-inf")})
        assert probs["a"] == 1.0
        assert probs["b"] == 0.0

    def test_all_inf_returns_uniform(self):
        probs = stable_softmax({"a": float("-inf"), "b": float("-inf")})
        assert probs["a"] == 0.5
        assert probs["b"] == 0.5

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            stable_softmax({})


# =========================================================================
# LabelTrie
# =========================================================================

class TestLabelTrie:
    def test_single_token_labels(self):
        trie = LabelTrie()
        trie.insert("positive", ["positive"])
        trie.insert("negative", ["negative"])
        trie.insert("neutral", ["neutral"])
        assert len(trie.root.children) == 3
        assert trie.root.children["positive"].is_terminal
        assert trie.root.children["positive"].label == "positive"

    def test_shared_prefix(self):
        trie = LabelTrie()
        trie.insert("account", ["acc", "ount"])
        trie.insert("access", ["acc", "ess"])
        assert len(trie.root.children) == 1
        acc = trie.root.children["acc"]
        assert not acc.is_terminal
        assert len(acc.children) == 2

    def test_max_branching_factor(self):
        trie = LabelTrie()
        trie.insert("positive", ["positive"])
        trie.insert("negative", ["negative"])
        trie.insert("neutral", ["neutral"])
        trie.insert("account", ["acc", "ount"])
        trie.insert("access", ["acc", "ess"])
        # Root has 4 children: positive, negative, neutral, acc
        # acc node has 2 children: ount, ess
        assert trie.max_branching_factor == 4

    def test_get_token_sequence(self):
        trie = LabelTrie()
        trie.insert("technical_support", ["techn", "ical", "_support"])
        assert trie.get_token_sequence("technical_support") == ["techn", "ical", "_support"]

    def test_all_labels(self):
        trie = LabelTrie()
        trie.insert("a", ["a"])
        trie.insert("b", ["b"])
        assert set(trie.all_labels()) == {"a", "b"}


# =========================================================================
# divergence_point
# =========================================================================

class TestDivergencePoint:
    def test_identical(self):
        assert divergence_point(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_first_token(self):
        assert divergence_point(["x", "b"], ["a", "b"]) == 0

    def test_middle(self):
        assert divergence_point(["a", "x", "c"], ["a", "b", "c"]) == 1

    def test_different_lengths_shorter_is_prefix(self):
        assert divergence_point(["a", "b"], ["a", "b", "c"]) == 2


# =========================================================================
# score_labels_from_winning_path
# =========================================================================

class TestScoreLabelsFromWinningPath:
    def test_case1_winner_a(self):
        """Winner=a, b diverges at token 3, c at token 0.
        All labels scored up to divergence point."""
        token_seqs = {
            "a": ["t1", "t2", "t3", "t4a"],
            "b": ["t1", "t2", "t3", "t4b"],
            "c": ["t1c", "t2c", "t3c", "t4c"],
        }
        step_lps = [
            {"t1": -0.1, "t1c": -2.5, "other": -5.0},
            {"t2": -0.2, "x": -3.0},
            {"t3": -0.15, "y": -2.8},
            {"t4a": -0.1, "t4b": -1.5, "z": -4.0},
        ]
        scores = score_labels_from_winning_path(token_seqs, "a", step_lps)
        assert abs(scores["a"] - ((-0.1 + -0.2 + -0.15 + -0.1) / 4)) < 1e-10
        assert abs(scores["b"] - ((-0.1 + -0.2 + -0.15 + -1.5) / 4)) < 1e-10
        assert abs(scores["c"] - (-2.5)) < 1e-10

    def test_case2_winner_c(self):
        """Winner=c, a and b diverge at token 0 but share prefix among themselves."""
        token_seqs = {
            "a": ["t1", "t2", "t3", "t4a"],
            "b": ["t1", "t2", "t3", "t4b"],
            "c": ["t1c", "t2c", "t3c", "t4c"],
        }
        step_lps = [
            {"t1": -2.0, "t1c": -0.1, "other": -5.0},
            {"t2c": -0.2, "x": -3.0},
            {"t3c": -0.15, "y": -2.8},
            {"t4c": -0.1, "z": -4.0},
        ]
        scores = score_labels_from_winning_path(token_seqs, "c", step_lps)
        assert abs(scores["a"] - (-2.0)) < 1e-10
        assert abs(scores["b"] - (-2.0)) < 1e-10
        assert abs(scores["c"] - ((-0.1 + -0.2 + -0.15 + -0.1) / 4)) < 1e-10

    def test_single_token_labels_exact(self):
        token_seqs = {"positive": ["positive"], "negative": ["negative"]}
        step_lps = [{"positive": -0.3, "negative": -1.5}]
        scores = score_labels_from_winning_path(token_seqs, "positive", step_lps)
        assert abs(scores["positive"] - (-0.3)) < 1e-10
        assert abs(scores["negative"] - (-1.5)) < 1e-10

    def test_missing_token_returns_neg_inf(self):
        token_seqs = {"rare": ["rare_tok"], "common": ["common_tok"]}
        step_lps = [{"common_tok": -0.1, "other": -1.0}]
        scores = score_labels_from_winning_path(token_seqs, "common", step_lps)
        assert scores["rare"] == float("-inf")
        assert scores["common"] == -0.1


# =========================================================================
# get_scored_lengths
# =========================================================================

class TestGetScoredLengths:
    def test_case1(self):
        token_seqs = {
            "a": ["t1", "t2", "t3", "t4a"],
            "b": ["t1", "t2", "t3", "t4b"],
            "c": ["t1c"],
        }
        lengths = get_scored_lengths(token_seqs, "a")
        assert lengths["a"] == 4
        assert lengths["b"] == 4  # diverges at 3, scored up to 3 inclusive = 4 tokens
        assert lengths["c"] == 1


# =========================================================================
# identify_unresolved_clusters
# =========================================================================

class TestIdentifyUnresolvedClusters:
    def test_case2_winner_c(self):
        """After scoring case 2 (winner=c), a and b form an unresolved cluster."""
        token_seqs = {
            "a": ["t1", "t2", "t3", "t4a"],
            "b": ["t1", "t2", "t3", "t4b"],
            "c": ["t1c", "t2c", "t3c", "t4c"],
        }
        scored_lengths = {"a": 1, "b": 1, "c": 4}
        clusters = identify_unresolved_clusters(token_seqs, scored_lengths)
        assert len(clusters) == 1
        assert set(clusters[0].labels) == {"a", "b"}
        assert clusters[0].resolved_length == 1

    def test_no_clusters_when_all_resolved(self):
        token_seqs = {"a": ["t1"], "b": ["t2"]}
        scored_lengths = {"a": 1, "b": 1}
        clusters = identify_unresolved_clusters(token_seqs, scored_lengths)
        assert len(clusters) == 0

    def test_multiple_clusters(self):
        token_seqs = {
            "a": ["t1", "t2"],
            "b": ["t1", "t3"],
            "c": ["t4", "t5"],
            "d": ["t4", "t6"],
        }
        scored_lengths = {"a": 1, "b": 1, "c": 1, "d": 1}
        clusters = identify_unresolved_clusters(token_seqs, scored_lengths)
        assert len(clusters) == 2

    def test_two_separate_clusters_different_prefixes(self):
        """Labels a/b share prefix t1; labels c/d share prefix t4.
        With winner=c, only a/b are unresolved as a pair."""
        token_seqs = {
            "a": ["t1", "t2"],
            "b": ["t1", "t3"],
            "c": ["t4", "t5"],
            "d": ["t4", "t6"],
        }
        # c is the winner, fully resolved; d shares prefix with c, resolved at token 0
        scored_lengths = {"a": 1, "b": 1, "c": 2, "d": 1}
        clusters = identify_unresolved_clusters(token_seqs, scored_lengths)
        # a and b share prefix t1 at resolved_length 1
        # d shares prefix t4 at resolved_length 1
        assert len(clusters) == 2
