"""Main classifier module for ollama-classifier."""

import json
import math
from typing import Union, List, Dict, Any

from ollama import Client, AsyncClient

from .types import ClassificationResult, ChoicesType
from .prompts import (
    build_classification_prompt,
    get_choice_labels,
    build_json_schema_for_choices,
)


class OllamaClassifier:
    """A classifier wrapper around Ollama client for text classification.
    
    This class provides methods for classifying text into a set of predefined choices
    with calibrated probability scores using multi-call evaluation.
    
    Attributes:
        _client: The Ollama client (sync or async).
        _model: The model name to use for classification.
    """
    
    def __init__(self, client: Union[Client, AsyncClient], model: str):
        """Initialize the classifier.
        
        Args:
            client: An initialized Ollama Client or AsyncClient instance.
            model: The model name to use for classification (e.g., "llama3.2").
        """
        self._client = client
        self._model = model
    
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
        
        Uses JSON schema with enum constraint to ensure only valid choices are generated.
        This is the fastest method as it only makes one API call and doesn't compute
        confidence scores.
        
        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            The predicted choice label.
        """
        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)
        schema = build_json_schema_for_choices(labels)
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        
        response = self._client.chat(
            model=self._model,
            messages=messages,
            format=schema,
            options={"temperature": 0.0},
        )
        
        result = json.loads(response.message.content)
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
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
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
        
        Makes separate API calls for each choice to compute log P(choice|context),
        then applies softmax for calibrated probabilities.
        This makes N API calls for N choices.
        
        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
        """
        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)
        
        # Compute logprob for each choice
        logprobs: Dict[str, float] = {}
        for label in labels:
            logprobs[label] = self._get_choice_logprob(
                system, user, label
            )
        
        # Apply softmax to get probabilities
        probabilities = self._softmax(logprobs)
        
        # Get prediction (highest probability)
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
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
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

        Uses multi-call evaluation to compute calibrated probabilities for each choice.
        Makes N API calls for N choices.

        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
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
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
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
        """Async version of generate().
        
        Generate a constrained classification for a single text.
        
        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            The predicted choice label.
        """
        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)
        schema = build_json_schema_for_choices(labels)
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        
        response = await self._client.chat(
            model=self._model,
            messages=messages,
            format=schema,
            options={"temperature": 0.0},
        )
        
        result = json.loads(response.message.content)
        return result.get("label", "")
    
    async def abatch_generate(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[str]:
        """Async version of batch_generate().
        
        Generate constrained classifications for multiple texts.
        
        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            List of predicted choice labels, one per input text.
        """
        import asyncio
        return await asyncio.gather(*[
            self.agenerate(text, choices, system_prompt) for text in texts
        ])
    
    # =========================================================================
    # Async Methods - Score
    # =========================================================================
    
    async def ascore(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Async version of score().
        
        Score a classification using multi-call evaluation with softmax.
        
        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
        """
        import asyncio
        
        labels = get_choice_labels(choices)
        system, user = build_classification_prompt(text, choices, system_prompt)
        
        # Compute logprob for each choice concurrently
        logprobs_tasks = [
            self._aget_choice_logprob(system, user, label)
            for label in labels
        ]
        logprob_values = await asyncio.gather(*logprobs_tasks)
        
        logprobs = dict(zip(labels, logprob_values))
        
        # Apply softmax to get probabilities
        probabilities = self._softmax(logprobs)
        
        # Get prediction (highest probability)
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
        """Async version of batch_score().
        
        Score multiple texts using multi-call method.
        
        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            List of ClassificationResults, one per input text.
        """
        import asyncio
        return await asyncio.gather(*[
            self.ascore(text, choices, system_prompt) for text in texts
        ])
    
    # =========================================================================
    # Async Methods - Classify
    # =========================================================================
    
    async def aclassify(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Async version of classify().

        Classify text with calibrated confidence scores using multi-call evaluation.

        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.

        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
        """
        return await self.ascore(text, choices, system_prompt)
    
    async def abatch_classify(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Async version of batch_classify().
        
        Classify multiple texts with calibrated confidence scores.
        
        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            List of ClassificationResults, one per input text.
        """
        import asyncio
        return await asyncio.gather(*[
            self.aclassify(text, choices, system_prompt) for text in texts
        ])
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _get_choice_logprob(
        self,
        system: str,
        user: str,
        choice: str,
    ) -> float:
        """Compute a log-probability score for a single choice.

        Generates a response with a single-value schema that forces the model
        to output ``choice`` and returns the sum of logprobs of all generated
        tokens.  This gives a score proportional to how naturally the model
        produces that choice given the context; applying softmax across all
        choices yields calibrated probabilities.

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

        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=forced_schema,
            logprobs=True,
            options={"temperature": 0.0},
        )

        if response.logprobs:
            return sum(lp.logprob for lp in response.logprobs)
        return 0.0

    async def _aget_choice_logprob(
        self,
        system: str,
        user: str,
        choice: str,
    ) -> float:
        """Async version of _get_choice_logprob().

        Compute a log-probability score for a single choice.

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

        response = await self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=forced_schema,
            logprobs=True,
            options={"temperature": 0.0},
        )

        if response.logprobs:
            return sum(lp.logprob for lp in response.logprobs)
        return 0.0

    def _softmax(self, logprobs: Dict[str, float]) -> Dict[str, float]:
        """Apply numerically stable softmax to log probabilities.
        
        Args:
            logprobs: Dict mapping labels to log probabilities.
            
        Returns:
            Dict mapping labels to probabilities (summing to 1.0).
        """
        # Filter out -inf values for computation
        valid_logprobs = {k: v for k, v in logprobs.items() if v > float("-inf")}
        
        if not valid_logprobs:
            # If all are -inf, return uniform distribution
            n = len(logprobs)
            return {k: 1.0 / n for k in logprobs}
        
        max_lp = max(valid_logprobs.values())
        exp_vals = {k: math.exp(v - max_lp) if v > float("-inf") else 0.0 
                    for k, v in logprobs.items()}
        total = sum(exp_vals.values())
        
        if total == 0:
            n = len(logprobs)
            return {k: 1.0 / n for k in logprobs}
        
        return {k: v / total for k, v in exp_vals.items()}