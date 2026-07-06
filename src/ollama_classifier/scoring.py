"""Probability and scoring utilities for classification.

All length normalization uses geometric mean (not arithmetic mean),
applied consistently across both generate() and classify() methods.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# =========================================================================
# Core probability functions
# =========================================================================

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

    Raises:
        ValueError: If logprobs is empty.
    """
    if not logprobs:
        raise ValueError("Cannot compute softmax of empty dict.")

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


# =========================================================================
# Trie data structure
# =========================================================================

@dataclass
class TrieNode:
    """A node in the label prefix trie."""
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    is_terminal: bool = False
    label: Optional[str] = None


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
        """Insert a label with its token sequence into the trie."""
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
        """Get the token sequence for a label."""
        return self._token_sequences[label]

    def all_labels(self) -> list[str]:
        """Get all labels in the trie."""
        return list(self._token_sequences.keys())


# =========================================================================
# Divergence-aware scoring
# =========================================================================

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
        prefix = tuple(seq[:resolved])
        clusters.setdefault(prefix, []).append(label)

    return [
        Cluster(labels=labels, resolved_length=len(prefix))
        for prefix, labels in clusters.items()
        if len(labels) > 0
    ]
