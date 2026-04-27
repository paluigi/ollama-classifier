"""Generic LLM classifier that works with any inference backend.

This module provides :class:`LLMClassifier`, a backend-agnostic classifier
that delegates inference to a :class:`~ollama_classifier.backends.base.LLMBackend`
instance.  The public API mirrors :class:`OllamaClassifier` so that switching
engines requires changing only the constructor.

Example usage with vLLM::

    from ollama_classifier.backends import VLLMBackend
    from ollama_classifier import LLMClassifier

    backend = VLLMBackend(model="meta-llama/Llama-3.2-3B-Instruct")
    classifier = LLMClassifier(backend)

    result = classifier.classify(
        text="I love this product!",
        choices=["positive", "negative", "neutral"],
    )
"""

import json
import math
from typing import Dict, List

from .backends.base import ChatMessage, ChatResponse, LLMBackend
from .prompts import (
    build_classification_prompt,
    build_json_schema_for_choices,
    get_choice_labels,
)
from .types import ClassificationResult, ChoicesType


class LLMClassifier:
    """A backend-agnostic text classifier.

    This class provides the same classification interface as
    :class:`OllamaClassifier` but delegates the actual LLM calls to any
    :class:`LLMBackend` implementation (vLLM, SGLang, llama.cpp, etc.).

    Attributes:
        _backend: The inference backend instance.
    """

    def __init__(self, backend: LLMBackend):
        """Initialize the classifier.

        Args:
            backend: An initialized LLMBackend instance (e.g.,
                     ``VLLMBackend(...)``, ``SGLangBackend(...)``, or
                     ``LlamaCppBackend(...)``).
        """
        self._backend = backend

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _to_messages(
        self, system: str, user: str
    ) -> List[ChatMessage]:
        """Convert prompt strings to a list of ChatMessage."""
        return [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]

    def _extract_logprob_sum(self, response: ChatResponse) -> float:
        """Extract the sum of token logprobs from a chat response."""
        if response.logprobs:
            return sum(entry.get("logprob", 0.0) for entry in response.logprobs)
        return 0.0

    def _softmax(self, logprobs: Dict[str, float]) -> Dict[str, float]:
        """Apply numerically stable softmax to log probabilities."""
        valid_logprobs = {k: v for k, v in logprobs.items() if v > float("-inf")}

        if not valid_logprobs:
            n = len(logprobs)
            return {k: 1.0 / n for k in logprobs}

        max_lp = max(valid_logprobs.values())
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
    # Sync Methods - Generate
    # =========================================================================

    def generate(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a constrained classification for a single text.

        Uses JSON schema with enum constraint to ensure only valid choices
        are generated.  This is the fastest method as it only makes one API
        call and does not compute confidence scores.

        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping
                     labels to descriptions.
            system_prompt: Optional custom system prompt.

        Returns:
            The predicted choice label.
        """
        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)
        schema = build_json_schema_for_choices(labels)
        messages = self._to_messages(system, user)

        response = self._backend.chat(
            messages=messages,
            temperature=0.0,
            guided_json=schema,
        )

        result = json.loads(response.content)
        return result.get("label", "")

    def batch_generate(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[str]:
        """Generate constrained classifications for multiple texts.

        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping
                     labels to descriptions.
            system_prompt: Optional custom system prompt.

        Returns:
            List of predicted choice labels, one per input text.
        """
        return [self.generate(text, choices, system_prompt) for text in texts]

    # =========================================================================
    # Sync Methods - Score (Multi-call evaluation with softmax)
    # =========================================================================

    def score(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Score a classification using multi-call evaluation with softmax.

        Makes separate API calls for each choice to compute
        log P(choice|context), then applies softmax for calibrated
        probabilities.  This makes N API calls for N choices.

        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping
                     labels to descriptions.
            system_prompt: Optional custom system prompt.

        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
        """
        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)

        logprobs: Dict[str, float] = {}
        for label in labels:
            logprobs[label] = self._get_choice_logprob(system, user, label)

        probabilities = self._softmax(logprobs)
        prediction = max(probabilities, key=probabilities.get)

        return ClassificationResult(
            prediction=prediction,
            confidence=probabilities[prediction],
            probabilities=probabilities,
            raw_response={"logprobs": logprobs},
        )

    def batch_score(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Score multiple texts using multi-call method.

        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping
                     labels to descriptions.
            system_prompt: Optional custom system prompt.

        Returns:
            List of ClassificationResults, one per input text.
        """
        return [self.score(text, choices, system_prompt) for text in texts]

    # =========================================================================
    # Sync Methods - Classify
    # =========================================================================

    def classify(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Classify text with calibrated confidence scores.

        Uses multi-call evaluation to compute calibrated probabilities
        for each choice.  Makes N API calls for N choices.

        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping
                     labels to descriptions.
            system_prompt: Optional custom system prompt.

        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
        """
        return self.score(text, choices, system_prompt)

    def batch_classify(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Classify multiple texts with calibrated confidence scores.

        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping
                     labels to descriptions.
            system_prompt: Optional custom system prompt.

        Returns:
            List of ClassificationResults, one per input text.
        """
        return [self.classify(text, choices, system_prompt) for text in texts]

    # =========================================================================
    # Async Methods - Generate
    # =========================================================================

    async def agenerate(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> str:
        """Async version of :meth:`generate`."""
        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)
        schema = build_json_schema_for_choices(labels)
        messages = self._to_messages(system, user)

        response = await self._backend.achat(
            messages=messages,
            temperature=0.0,
            guided_json=schema,
        )

        result = json.loads(response.content)
        return result.get("label", "")

    async def abatch_generate(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[str]:
        """Async version of :meth:`batch_generate`."""
        import asyncio

        return await asyncio.gather(
            *[self.agenerate(text, choices, system_prompt) for text in texts]
        )

    # =========================================================================
    # Async Methods - Score
    # =========================================================================

    async def ascore(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Async version of :meth:`score`."""
        import asyncio

        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)

        logprobs_tasks = [
            self._aget_choice_logprob(system, user, label) for label in labels
        ]
        logprob_values = await asyncio.gather(*logprobs_tasks)
        logprobs = dict(zip(labels, logprob_values))

        probabilities = self._softmax(logprobs)
        prediction = max(probabilities, key=probabilities.get)

        return ClassificationResult(
            prediction=prediction,
            confidence=probabilities[prediction],
            probabilities=probabilities,
            raw_response={"logprobs": logprobs},
        )

    async def abatch_score(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Async version of :meth:`batch_score`."""
        import asyncio

        return await asyncio.gather(
            *[self.ascore(text, choices, system_prompt) for text in texts]
        )

    # =========================================================================
    # Async Methods - Classify
    # =========================================================================

    async def aclassify(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Async version of :meth:`classify`."""
        return await self.ascore(text, choices, system_prompt)

    async def abatch_classify(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Async version of :meth:`batch_classify`."""
        import asyncio

        return await asyncio.gather(
            *[self.aclassify(text, choices, system_prompt) for text in texts]
        )

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _get_choice_logprob(
        self,
        system: str,
        user: str,
        choice: str,
    ) -> float:
        """Compute a log-probability score for a single choice.

        Forces the model to output *choice* and returns the sum of
        logprobs of all generated tokens.

        Args:
            system: System prompt.
            user: User prompt.
            choice: The choice label to evaluate.

        Returns:
            Sum of token logprobs for the forced-choice generation.
        """
        forced_schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": [choice]},
            },
            "required": ["label"],
        }

        messages = self._to_messages(system, user)
        response = self._backend.chat(
            messages=messages,
            temperature=0.0,
            guided_json=forced_schema,
            logprobs=True,
        )

        return self._extract_logprob_sum(response)

    async def _aget_choice_logprob(
        self,
        system: str,
        user: str,
        choice: str,
    ) -> float:
        """Async version of :meth:`_get_choice_logprob`."""
        forced_schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": [choice]},
            },
            "required": ["label"],
        }

        messages = self._to_messages(system, user)
        response = await self._backend.achat(
            messages=messages,
            temperature=0.0,
            guided_json=forced_schema,
            logprobs=True,
        )

        return self._extract_logprob_sum(response)
