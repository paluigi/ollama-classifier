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
    with support for both fast (single-call) and complete (multi-call) scoring methods.
    
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
        
        content = response.get("message", {}).get("content", "{}")
        result = json.loads(content)
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
    # Sync Methods - Score Fast (Single-call logprob extraction)
    # =========================================================================
    
    def score_fast(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Score a classification using single-call logprob extraction.
        
        Makes one API call with logprobs enabled and extracts the probability
        distribution from the token distribution at the decision point.
        This is faster but may be less reliable for capturing all choice probabilities.
        
        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
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
            logprobs=True,
            top_logprobs=20,
            options={"temperature": 0.0},
        )
        
        # Extract prediction from response
        content = response.get("message", {}).get("content", "{}")
        result = json.loads(content)
        prediction = result.get("label", "")
        
        # Extract logprobs from token distribution
        probabilities = self._extract_probabilities_from_logprobs(
            response, labels
        )
        
        return ClassificationResult(
            prediction=prediction,
            confidence=probabilities.get(prediction, 0.0),
            probabilities=probabilities,
            raw_response=response,
        )
    
    def batch_score_fast(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Score multiple texts using fast single-call method.
        
        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            List of ClassificationResults, one per input text.
        """
        return [self.score_fast(text, choices, system_prompt) for text in texts]
    
    # =========================================================================
    # Sync Methods - Score Complete (Multi-call evaluation)
    # =========================================================================
    
    def score_complete(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Score a classification using multi-call evaluation with softmax.
        
        Makes separate API calls for each choice to compute log P(choice|context),
        then applies softmax for calibrated probabilities.
        This is more accurate but slower (N API calls for N choices).
        
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
    
    def batch_score_complete(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Score multiple texts using complete multi-call method.
        
        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            List of ClassificationResults, one per input text.
        """
        return [self.score_complete(text, choices, system_prompt) for text in texts]
    
    # =========================================================================
    # Sync Methods - Classify (Generate + Score)
    # =========================================================================
    
    def classify(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Classify text with constrained generation and fast scoring.
        
        Combines generate() and score_fast() for a single classification
        with confidence scores.
        
        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
        """
        return self.score_fast(text, choices, system_prompt)
    
    def classify_complete(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Classify text with constrained generation and complete scoring.
        
        Combines generate() and score_complete() for accurate classification
        with calibrated confidence scores.
        
        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
        """
        # Use generate for constrained prediction
        prediction = self.generate(text, choices, system_prompt)
        
        # Use score_complete for probabilities
        result = self.score_complete(text, choices, system_prompt)
        
        # Override prediction with constrained output
        result.prediction = prediction
        result.confidence = result.probabilities.get(prediction, 0.0)
        
        return result
    
    def batch_classify(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Classify multiple texts with fast scoring.
        
        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            List of ClassificationResults, one per input text.
        """
        return [self.classify(text, choices, system_prompt) for text in texts]
    
    def batch_classify_complete(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Classify multiple texts with complete scoring.
        
        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            List of ClassificationResults, one per input text.
        """
        return [self.classify_complete(text, choices, system_prompt) for text in texts]
    
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
        
        content = response.get("message", {}).get("content", "{}")
        result = json.loads(content)
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
    # Async Methods - Score Fast
    # =========================================================================
    
    async def ascore_fast(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Async version of score_fast().
        
        Score a classification using single-call logprob extraction.
        
        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
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
            logprobs=True,
            top_logprobs=20,
            options={"temperature": 0.0},
        )
        
        # Extract prediction from response
        content = response.get("message", {}).get("content", "{}")
        result = json.loads(content)
        prediction = result.get("label", "")
        
        # Extract logprobs from token distribution
        probabilities = self._extract_probabilities_from_logprobs(
            response, labels
        )
        
        return ClassificationResult(
            prediction=prediction,
            confidence=probabilities.get(prediction, 0.0),
            probabilities=probabilities,
            raw_response=response,
        )
    
    async def abatch_score_fast(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Async version of batch_score_fast().
        
        Score multiple texts using fast single-call method.
        
        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            List of ClassificationResults, one per input text.
        """
        import asyncio
        return await asyncio.gather(*[
            self.ascore_fast(text, choices, system_prompt) for text in texts
        ])
    
    # =========================================================================
    # Async Methods - Score Complete
    # =========================================================================
    
    async def ascore_complete(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Async version of score_complete().
        
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
    
    async def abatch_score_complete(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Async version of batch_score_complete().
        
        Score multiple texts using complete multi-call method.
        
        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            List of ClassificationResults, one per input text.
        """
        import asyncio
        return await asyncio.gather(*[
            self.ascore_complete(text, choices, system_prompt) for text in texts
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
        
        Classify text with constrained generation and fast scoring.
        
        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
        """
        return await self.ascore_fast(text, choices, system_prompt)
    
    async def aclassify_complete(
        self,
        text: str,
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> ClassificationResult:
        """Async version of classify_complete().
        
        Classify text with constrained generation and complete scoring.
        
        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
        """
        # Use generate for constrained prediction
        prediction = await self.agenerate(text, choices, system_prompt)
        
        # Use score_complete for probabilities
        result = await self.ascore_complete(text, choices, system_prompt)
        
        # Override prediction with constrained output
        result.prediction = prediction
        result.confidence = result.probabilities.get(prediction, 0.0)
        
        return result
    
    async def abatch_classify(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Async version of batch_classify().
        
        Classify multiple texts with fast scoring.
        
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
    
    async def abatch_classify_complete(
        self,
        texts: List[str],
        choices: ChoicesType,
        system_prompt: str | None = None,
    ) -> List[ClassificationResult]:
        """Async version of batch_classify_complete().
        
        Classify multiple texts with complete scoring.
        
        Args:
            texts: List of texts to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.
            
        Returns:
            List of ClassificationResults, one per input text.
        """
        import asyncio
        return await asyncio.gather(*[
            self.aclassify_complete(text, choices, system_prompt) for text in texts
        ])
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _extract_probabilities_from_logprobs(
        self,
        response: Dict[str, Any],
        labels: List[str],
    ) -> Dict[str, float]:
        """Extract probabilities from logprobs in the response.
        
        Looks for the decision token in the logprobs and extracts
        the probability distribution for the valid choices.
        
        Args:
            response: The raw API response.
            labels: List of valid choice labels.
            
        Returns:
            Dict mapping choice labels to probabilities.
        """
        # Initialize with zero logprobs
        choice_logprobs: Dict[str, float] = {label: float("-inf") for label in labels}
        
        # Navigate the logprobs payload to find the decision token
        logprobs_data = response.get("logprobs", [])
        
        if not logprobs_data:
            # No logprobs available, return uniform distribution
            return {label: 1.0 / len(labels) for label in labels}
        
        for token_data in logprobs_data:
            # Check if this is a dict or list format
            if isinstance(token_data, dict):
                top_candidates = token_data.get("top_logprobs", [])
            elif isinstance(token_data, (list, tuple)) and len(token_data) > 0:
                # Might be in different format
                top_candidates = token_data if isinstance(token_data[0], dict) else []
            else:
                continue
            
            found_decision_token = False
            
            for candidate in top_candidates:
                if isinstance(candidate, dict):
                    token = candidate.get("token", "")
                    # Normalize the token
                    clean_token = token.strip(' \n\r\t"\'').lower()
                    
                    # Check if this matches any of our labels
                    for label in labels:
                        if label.lower() == clean_token or clean_token in label.lower():
                            logprob = candidate.get("logprob", float("-inf"))
                            if logprob > choice_logprobs[label]:
                                choice_logprobs[label] = logprob
                            found_decision_token = True
            
            if found_decision_token:
                break
        
        # Apply softmax to convert logprobs to probabilities
        return self._softmax(choice_logprobs)
    
    def _get_choice_logprob(
        self,
        system: str,
        user: str,
        choice: str,
    ) -> float:
        """Compute log P(choice | context) for a single choice.
        
        Appends the choice as assistant response and measures
        how "expected" those tokens were.
        
        Args:
            system: System prompt.
            user: User prompt.
            choice: The choice label to evaluate.
            
        Returns:
            Log probability of the choice given the context.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps({"label": choice})},
        ]
        
        response = self._client.chat(
            model=self._model,
            messages=messages,
            logprobs=True,
            options={"num_predict": 0},
        )
        
        return self._extract_logprob_from_response(response)
    
    async def _aget_choice_logprob(
        self,
        system: str,
        user: str,
        choice: str,
    ) -> float:
        """Async version of _get_choice_logprob().
        
        Compute log P(choice | context) for a single choice.
        
        Args:
            system: System prompt.
            user: User prompt.
            choice: The choice label to evaluate.
            
        Returns:
            Log probability of the choice given the context.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps({"label": choice})},
        ]
        
        response = await self._client.chat(
            model=self._model,
            messages=messages,
            logprobs=True,
            options={"num_predict": 0},
        )
        
        return self._extract_logprob_from_response(response)
    
    def _extract_logprob_from_response(self, response: Dict[str, Any]) -> float:
        """Extract log probability from Ollama response.
        
        Tries various fields where Ollama might return logprobs.
        
        Args:
            response: The raw API response.
            
        Returns:
            Extracted log probability, or 0.0 if not found.
        """
        msg = response.get("message", {})
        
        # Field: message.logprob (most common in newer versions)
        if "logprob" in msg:
            return msg["logprob"]
        
        # Field: eval_prob
        if "eval_prob" in response:
            p = response["eval_prob"]
            return math.log(p) if p > 0 else float("-inf")
        
        # Field: logprobs list
        if "logprobs" in msg:
            lp = msg["logprobs"]
            if isinstance(lp, list) and lp:
                # Sum or average the logprobs
                return sum(lp) / len(lp)
        
        # Check in response logprobs
        logprobs = response.get("logprobs", [])
        if logprobs and isinstance(logprobs, list):
            # Try to extract from first token's logprob
            if isinstance(logprobs[0], dict) and "logprob" in logprobs[0]:
                return logprobs[0]["logprob"]
        
        return 0.0  # Default neutral
    
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