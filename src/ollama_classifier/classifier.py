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
        result = json.loads(response.message.content)
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
        """Classify text with complete scoring (N API calls, calibrated probabilities).

        Uses score_complete() which makes one forced-choice API call per label,
        then picks the highest-probability label as the prediction.

        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.

        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
        """
        return self.score_complete(text, choices, system_prompt)
    
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
        result = json.loads(response.message.content)
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

        Classify text with complete scoring (N concurrent API calls, calibrated
        probabilities). Picks the highest-probability label as the prediction.

        Args:
            text: The text to classify.
            choices: Either a list of choice labels, or a dict mapping labels to descriptions.
            system_prompt: Optional custom system prompt.

        Returns:
            ClassificationResult with prediction, confidence, and probabilities.
        """
        return await self.ascore_complete(text, choices, system_prompt)
    
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
        response: Any,
        labels: List[str],
    ) -> Dict[str, float]:
        """Extract probabilities from logprobs in the response.

        Locates the token position where the label value begins in the generated
        JSON output, then walks forward position by position accumulating the
        log-probability for each label until every label has been fully resolved.
        At each position the best-matching token for each label is looked up in
        ``top_logprobs``; its logprob is added to that label's running score and
        the matched character count is advanced.  This continues until all labels
        are fully consumed, ensuring that labels sharing a long common prefix are
        always unambiguously distinguished.

        Args:
            response: The raw ChatResponse object returned by the Ollama client.
            labels: List of valid choice labels.

        Returns:
            Dict mapping choice labels to probabilities (summing to 1.0).
        """
        logprobs_data = response.logprobs  # List[Logprob] | None

        if not logprobs_data:
            # No logprobs available – return uniform distribution
            return {label: 1.0 / len(labels) for label in labels}

        # ------------------------------------------------------------------
        # Step 1: find the start position.
        #
        # The model generates JSON of the form `{ "label": "CHOICE" }`.
        # We scan for the first token whose text (stripped of surrounding
        # spaces and quote characters) is a non-empty prefix of at least one
        # label.  For the example above this lands on the token `b` (start of
        # "bullish"), skipping the structural tokens `{`, ` "`, `label`, `":`,
        # and ` "`.
        # ------------------------------------------------------------------
        start_idx: int | None = None
        for i, lp in enumerate(logprobs_data):
            clean = lp.token.strip(' "\n\r\t')
            if clean and any(
                label.lower().startswith(clean.lower()) for label in labels
            ):
                start_idx = i
                break

        if start_idx is None:
            # Could not locate the decision point – fall back to uniform
            return {label: 1.0 / len(labels) for label in labels}

        # ------------------------------------------------------------------
        # Step 2: walk forward from start_idx accumulating per-label scores.
        #
        # For each label we track:
        #   - ``consumed``: how many characters of the label have been matched
        #   - ``score``:    sum of logprobs of matched tokens (starts at 0.0)
        #
        # At every position we build a token→logprob lookup from top_logprobs
        # (plus the actually-generated token, which may be absent from the
        # top list).  For each still-unresolved label we find the longest
        # token in that lookup whose stripped text is a prefix of the label's
        # remaining characters, add its logprob to the label's score, and
        # advance ``consumed`` accordingly.
        #
        # We stop as soon as every label has been fully consumed, or when we
        # run out of token positions.
        # ------------------------------------------------------------------
        label_score: Dict[str, float] = {label: 0.0 for label in labels}
        label_consumed: Dict[str, int] = {label: 0 for label in labels}
        label_matched: Dict[str, bool] = {label: False for label in labels}

        for i in range(start_idx, len(logprobs_data)):
            lp = logprobs_data[i]

            # Build token → logprob map; ensure the generated token is present
            token_map: Dict[str, float] = {
                t.token: t.logprob for t in (lp.top_logprobs or [])
            }
            token_map[lp.token] = lp.logprob

            all_done = True
            for label in labels:
                consumed = label_consumed[label]
                if consumed >= len(label):
                    continue  # this label is already fully matched
                all_done = False

                remaining = label[consumed:]  # characters yet to be matched

                # Find the longest token in token_map that is a prefix of
                # `remaining` (case-insensitive after stripping whitespace and
                # quote characters).
                best_logprob: float = float("-inf")
                best_len: int = 0

                for token, logprob in token_map.items():
                    clean_token = token.strip(' "\n\r\t')
                    if not clean_token:
                        continue
                    if remaining.lower().startswith(clean_token.lower()):
                        if len(clean_token) > best_len:
                            best_logprob = logprob
                            best_len = len(clean_token)

                if best_len > 0:
                    label_score[label] += best_logprob
                    label_matched[label] = True
                    label_consumed[label] += best_len

            if all_done:
                break

        # Labels that never matched any token get -inf so softmax assigns them
        # zero probability rather than the spuriously high weight of exp(0).
        for label in labels:
            if not label_matched[label]:
                label_score[label] = float("-inf")

        # Apply softmax to convert accumulated log-scores to probabilities
        return self._softmax(label_score)

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
