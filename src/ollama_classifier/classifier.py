"""Unified LLM classifier with adaptive scoring.

Provides :class:`LLMClassifier`, a backend-agnostic classifier with two
confidence scoring methods:

- **``generate()``** — Hierarchical constrained generation.  A single
  constrained call produces a probability distribution over all labels using
  divergence-aware logprobs from the winning path.  When ``max_calls > 1``,
  supplementary calls resolve clusters of labels that share a token prefix
  but diverge from the winner — but only to **reproportion** probability mass
  *within* each cluster, never changing the between-group totals.  This
  guarantees accuracy never degrades as the call budget grows.

- **``classify()`` — Multi-call completion scoring with geometric-mean
  normalization.  Always exact.  Makes N calls for N labels (parallelizable
  via async).

Example::

    from ollama_classifier import LLMClassifier
    from ollama_classifier.backends import OllamaBackend

    backend = OllamaBackend(model="llama3.2")
    classifier = LLMClassifier(backend)

    result = classifier.classify(
        text="I love this product!",
        choices=["positive", "negative", "neutral"],
    )
    print(f"Prediction: {result.prediction} ({result.confidence:.2%})")
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from .backends.base import ChatMessage, ChatResponse, LLMBackend, TokenLogprob
from .prompts import build_classification_prompt, get_choice_labels
from .scoring import (
    Cluster,
    LabelTrie,
    divergence_point,
    geometric_mean_logprob,
    get_scored_lengths,
    identify_unresolved_clusters,
    score_labels_from_winning_path,
    stable_softmax,
)
from .types import ClassificationResult, ChoicesType


class LLMClassifier:
    """Backend-agnostic text classifier with two confidence scoring methods.

    Methods:
        generate(): Adaptive constrained generation with divergence-aware
            confidence. Budget-controlled via ``max_calls``.
        classify(): Multi-call completion scoring with geometric-mean
            normalization. Always exact.

    Args:
        backend: An ``LLMBackend`` instance (e.g., ``OllamaBackend``,
            ``VLLMBackend``, ``SGLangBackend``, or ``LlamaCppBackend``).
        max_workers: Thread pool size for sync batch operations.
    """

    def __init__(self, backend: LLMBackend, *, max_workers: int = 4):
        self._backend = backend
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # ==================================================================
    # generate() — Hierarchical constrained generation
    # ==================================================================

    def generate(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
        *,
        max_calls: int | None = 1,
    ) -> ClassificationResult:
        """Hierarchical constrained classification with divergence-aware confidence.

        Makes 1 to ``max_calls`` constrained API calls.

        **Call 1** constrains the model to all labels and returns
        ``top_logprobs`` at every generated position.  Labels are scored up to
        their divergence point from the winning path using the geometric mean
        of available token logprobs, then a softmax produces the initial
        probability distribution.  All logprobs come from the same constraint
        context, so the distribution is internally consistent.

        **Calls 2…max_calls** resolve *clusters*: groups of ≥2 non-winning
        labels that share a scored prefix but diverge from the winner (and from
        each other) at a later position.  For each cluster, a constrained call
        over only the cluster's labels produces divergence-based logprobs whose
        softmax gives **relative weights** summing to 1.  The cluster's total
        probability mass (summed from the initial distribution) is then
        redistributed among its members according to these relative weights.
        This **reproportioning** never changes the total probability of any
        group — it only sharpens the distribution *within* a group — so
        accuracy can only improve or stay the same, never degrade.

        With ``max_calls=1``: single call, partial scoring (fast, approximate).
        With ``max_calls=None``: resolves all clusters recursively (exact for
        non-winning labels; equivalent to ``classify()`` when every label is
        fully resolved).

        Args:
            text: Text to classify.
            choices: Labels as list or ``{label: description}`` dict.
            system_prompt: Optional custom system prompt.
            max_calls: Maximum number of API calls. ``None`` = unlimited.
                ``1`` (default) = single call, no cluster resolution.

        Returns:
            ``ClassificationResult`` with ``method="adaptive_generate"``.
        """
        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]

        # 1. Tokenize labels in the backend's constraint context
        token_context = self._get_token_context()
        token_sequences = self._tokenize_labels(labels, token_context)

        # 2. Build trie and determine required top_logprobs K
        trie = LabelTrie()
        for label, tokens in token_sequences.items():
            trie.insert(label, tokens)
        k = max(trie.max_branching_factor, 5)

        # 3. First constrained call over ALL labels
        response = self._backend.chat(
            messages=messages,
            temperature=0.0,
            constrain_labels=list(labels),
            logprobs=True,
            top_logprobs=k,
        )
        calls_made = 1

        step_lps = self._extract_step_logprobs(
            response, token_sequences, list(labels)
        )

        winning_label = response.label
        cluster_scores = score_labels_from_winning_path(
            token_sequences, winning_label, step_lps
        )
        cluster_lengths = get_scored_lengths(token_sequences, winning_label)

        # Accumulate per-label logprobs and coverage
        all_step_logprobs: dict[str, list[float]] = {}
        all_scored_lengths: dict[str, int] = {}
        for label in labels:
            scored_len = cluster_lengths[label]
            lps: list[float] = []
            for i in range(scored_len):
                token = token_sequences[label][i]
                if i < len(step_lps) and token in step_lps[i]:
                    lps.append(step_lps[i][token])
                else:
                    lps.append(float("-inf"))
            all_step_logprobs[label] = lps
            all_scored_lengths[label] = scored_len

        # 4. Initial probability distribution (all from same constraint context)
        raw_scores: dict[str, float] = {}
        for label in labels:
            lps = all_step_logprobs[label]
            raw_scores[label] = geometric_mean_logprob(lps) if lps else float("-inf")

        probabilities = stable_softmax(raw_scores)

        # 5. Recursive cluster resolution via reproportioning
        #
        # Identify clusters of ≥2 labels that share a scored prefix but diverge
        # from the winner.  For each cluster, make a constrained call over only
        # the cluster's labels, compute divergence-based relative weights (softmax
        # of geometric-mean scores), and redistribute the cluster's total
        # probability mass accordingly.  Between-group probabilities are locked.
        frontier: list[Cluster] = identify_unresolved_clusters(
            token_sequences, all_scored_lengths
        )

        while frontier and (max_calls is None or calls_made < max_calls):
            cluster = frontier.pop(0)
            cluster_labels = cluster.labels

            # Only resolve clusters with ≥2 labels (singletons are already fixed)
            if len(cluster_labels) < 2:
                continue

            resolved_len = cluster.resolved_length

            # Constrained call over only this cluster's labels
            cluster_response = self._backend.chat(
                messages=messages,
                temperature=0.0,
                constrain_labels=cluster_labels,
                logprobs=True,
                top_logprobs=k,
            )
            calls_made += 1

            cluster_step_lps = self._extract_step_logprobs(
                cluster_response, token_sequences, cluster_labels
            )

            # Score cluster labels from the subset call
            cluster_winner = cluster_response.label
            cluster_token_seqs = {
                l: token_sequences[l] for l in cluster_labels
            }
            sub_scores = score_labels_from_winning_path(
                cluster_token_seqs, cluster_winner, cluster_step_lps
            )
            sub_lengths = get_scored_lengths(
                cluster_token_seqs, cluster_winner
            )

            # Update per-label logprobs for cluster members
            for label in cluster_labels:
                new_len = sub_lengths[label]
                if new_len > all_scored_lengths[label]:
                    new_lps: list[float] = []
                    for i in range(new_len):
                        token = token_sequences[label][i]
                        if i < len(cluster_step_lps) and token in cluster_step_lps[i]:
                            new_lps.append(cluster_step_lps[i][token])
                        else:
                            new_lps.append(float("-inf"))
                    all_step_logprobs[label] = new_lps
                    all_scored_lengths[label] = new_len

            # Reproportion: redistribute cluster's total probability mass
            cluster_total = sum(probabilities[l] for l in cluster_labels)

            cluster_raw: dict[str, float] = {}
            for label in cluster_labels:
                lps = all_step_logprobs[label]
                cluster_raw[label] = (
                    geometric_mean_logprob(lps) if lps else float("-inf")
                )
            cluster_weights = stable_softmax(cluster_raw)

            for label in cluster_labels:
                probabilities[label] = cluster_total * cluster_weights[label]

            # Identify sub-clusters within this cluster for further resolution
            sub_clusters = identify_unresolved_clusters(
                cluster_token_seqs, sub_lengths
            )
            frontier.extend(sub_clusters)

        # 6. Compute coverage and final values
        coverage: dict[str, float] = {}
        for label in labels:
            total = len(token_sequences[label])
            scored = all_scored_lengths[label]
            coverage[label] = scored / total if total > 0 else 1.0

        is_approximate = any(c < 1.0 for c in coverage.values())
        prediction = max(probabilities, key=probabilities.get)

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
        context is ``None`` — labels are tokenized standalone.

        For Ollama (JSON enum wrapper), context is the JSON prefix that
        precedes the label in the response: ``'{"label": "'``.
        """
        if self._backend.supports_bare_label_constraint:
            return None
        else:
            return '{"label": "'

    def _tokenize_labels(
        self,
        labels: list[str],
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
        response: ChatResponse,
        token_sequences: dict[str, list[str]],
        cluster_labels: list[str],
    ) -> list[dict[str, float]]:
        """Extract per-step top_logprobs from a constrained call response.

        For bare-label backends, the response contains only label tokens.
        For Ollama (JSON wrapper), structural tokens are filtered by matching
        against known label tokens.
        """
        if not response.logprobs:
            return []

        # Collect all valid label tokens for filtering
        valid_tokens: set[str] = set()
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
    # classify() — Multi-call completion scoring
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
        normalization to eliminate token-count bias. This is the gold-standard
        confidence method.

        Makes N API calls for N choices (parallelizable via async).

        Args:
            text: Text to classify.
            choices: Labels as list or ``{label: description}`` dict.
            system_prompt: Optional custom system prompt.

        Returns:
            ``ClassificationResult`` with ``method="multi_call"``,
            ``approximate=False``.
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
            raw_response={
                "logprobs": raw_scores,
                "token_logprobs": logprob_details,
            },
        )

    # ==================================================================
    # Batch methods (parallelized)
    # ==================================================================

    def batch_generate(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
        *,
        max_calls: int | None = 1,
    ) -> List[ClassificationResult]:
        """Batch adaptive generation (parallelized via ``ThreadPoolExecutor``)."""
        return list(
            self._executor.map(
                lambda t: self.generate(t, choices, system_prompt, max_calls=max_calls),
                texts,
            )
        )

    def batch_classify(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Batch multi-call classification (parallelized via ``ThreadPoolExecutor``)."""
        return list(
            self._executor.map(
                lambda t: self.classify(t, choices, system_prompt), texts
            )
        )

    # ==================================================================
    # Async methods
    # ==================================================================

    async def agenerate(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
        *,
        max_calls: int | None = 1,
    ) -> ClassificationResult:
        """Async hierarchical constrained generation.

        See :meth:`generate` for the full algorithm description.
        """
        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]

        token_context = self._get_token_context()

        # Tokenize labels concurrently
        token_tasks = [
            self._backend.atokenize(label, context=token_context) for label in labels
        ]
        token_results = await asyncio.gather(*token_tasks)
        token_sequences = {
            label: [t.text if t.text else f"token_{t.id}" for t in tokens]
            for label, tokens in zip(labels, token_results)
        }

        trie = LabelTrie()
        for label, tokens in token_sequences.items():
            trie.insert(label, tokens)
        k = max(trie.max_branching_factor, 5)

        # 3. First constrained call over ALL labels
        response = await self._backend.achat(
            messages=messages,
            temperature=0.0,
            constrain_labels=list(labels),
            logprobs=True,
            top_logprobs=k,
        )
        calls_made = 1

        step_lps = self._extract_step_logprobs(
            response, token_sequences, list(labels)
        )

        winning_label = response.label
        cluster_scores = score_labels_from_winning_path(
            token_sequences, winning_label, step_lps
        )
        cluster_lengths = get_scored_lengths(token_sequences, winning_label)

        # Accumulate per-label logprobs and coverage
        all_step_logprobs: dict[str, list[float]] = {}
        all_scored_lengths: dict[str, int] = {}
        for label in labels:
            scored_len = cluster_lengths[label]
            lps: list[float] = []
            for i in range(scored_len):
                token = token_sequences[label][i]
                if i < len(step_lps) and token in step_lps[i]:
                    lps.append(step_lps[i][token])
                else:
                    lps.append(float("-inf"))
            all_step_logprobs[label] = lps
            all_scored_lengths[label] = scored_len

        # 4. Initial probability distribution
        raw_scores: dict[str, float] = {}
        for label in labels:
            lps = all_step_logprobs[label]
            raw_scores[label] = geometric_mean_logprob(lps) if lps else float("-inf")

        probabilities = stable_softmax(raw_scores)

        # 5. Recursive cluster resolution via reproportioning
        frontier: list[Cluster] = identify_unresolved_clusters(
            token_sequences, all_scored_lengths
        )

        while frontier and (max_calls is None or calls_made < max_calls):
            cluster = frontier.pop(0)
            cluster_labels = cluster.labels

            if len(cluster_labels) < 2:
                continue

            cluster_response = await self._backend.achat(
                messages=messages,
                temperature=0.0,
                constrain_labels=cluster_labels,
                logprobs=True,
                top_logprobs=k,
            )
            calls_made += 1

            cluster_step_lps = self._extract_step_logprobs(
                cluster_response, token_sequences, cluster_labels
            )

            cluster_winner = cluster_response.label
            cluster_token_seqs = {
                l: token_sequences[l] for l in cluster_labels
            }
            sub_scores = score_labels_from_winning_path(
                cluster_token_seqs, cluster_winner, cluster_step_lps
            )
            sub_lengths = get_scored_lengths(
                cluster_token_seqs, cluster_winner
            )

            for label in cluster_labels:
                new_len = sub_lengths[label]
                if new_len > all_scored_lengths[label]:
                    new_lps: list[float] = []
                    for i in range(new_len):
                        token = token_sequences[label][i]
                        if i < len(cluster_step_lps) and token in cluster_step_lps[i]:
                            new_lps.append(cluster_step_lps[i][token])
                        else:
                            new_lps.append(float("-inf"))
                    all_step_logprobs[label] = new_lps
                    all_scored_lengths[label] = new_len

            # Reproportion: redistribute cluster's total probability mass
            cluster_total = sum(probabilities[l] for l in cluster_labels)

            cluster_raw: dict[str, float] = {}
            for label in cluster_labels:
                lps = all_step_logprobs[label]
                cluster_raw[label] = (
                    geometric_mean_logprob(lps) if lps else float("-inf")
                )
            cluster_weights = stable_softmax(cluster_raw)

            for label in cluster_labels:
                probabilities[label] = cluster_total * cluster_weights[label]

            sub_clusters = identify_unresolved_clusters(
                cluster_token_seqs, sub_lengths
            )
            frontier.extend(sub_clusters)

        # 6. Compute coverage and final values
        coverage: dict[str, float] = {}
        for label in labels:
            total = len(token_sequences[label])
            scored = all_scored_lengths[label]
            coverage[label] = scored / total if total > 0 else 1.0

        is_approximate = any(c < 1.0 for c in coverage.values())
        prediction = max(probabilities, key=probabilities.get)

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

    async def aclassify(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
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
        logprob_details: dict[str, list[float]] = {}
        for label, scoring in zip(labels, scoring_results):
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
            raw_response={
                "logprobs": raw_scores,
                "token_logprobs": logprob_details,
            },
        )

    async def abatch_generate(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
        *,
        max_calls: int | None = 1,
    ) -> List[ClassificationResult]:
        """Async batch adaptive generation."""
        return await asyncio.gather(
            *[
                self.agenerate(t, choices, system_prompt, max_calls=max_calls)
                for t in texts
            ]
        )

    async def abatch_classify(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Async batch multi-call classification."""
        return await asyncio.gather(
            *[self.aclassify(t, choices, system_prompt) for t in texts]
        )
