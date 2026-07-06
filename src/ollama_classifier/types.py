"""Type definitions for ollama-classifier."""

from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    """Result of a classification operation.

    Attributes:
        prediction: The predicted class label.
        confidence: Confidence score for the prediction (0.0 to 1.0).
        probabilities: Probability distribution over all choices.
        method: Scoring method used: ``"adaptive_generate"`` or ``"multi_call"``.
        approximate: True if any label has partial coverage (unresolved tokens).
            Only relevant for ``adaptive_generate``; always ``False`` for
            ``multi_call``.
        coverage: Per-label fraction of tokens scored (0.0 to 1.0).
            ``1.0`` = fully resolved. Only present for ``adaptive_generate``.
        n_calls: Number of API calls made.
        raw_response: Raw data for debugging.
    """

    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    method: str = "multi_call"
    approximate: bool = False
    coverage: Dict[str, float] = Field(default_factory=dict)
    n_calls: int = 1
    raw_response: Dict[str, Any] = Field(default_factory=dict)


# Type alias for choices — list of labels or dict mapping labels to descriptions
ChoicesType = Union[List[str], Dict[str, str]]
