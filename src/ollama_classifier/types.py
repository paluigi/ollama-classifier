"""Type definitions for ollama-classifier."""

from dataclasses import dataclass, field
from typing import Dict, Any, Union, List


@dataclass
class ClassificationResult:
    """Result of a classification operation.

    Attributes:
        prediction: The predicted class label.
        confidence: Confidence score for the prediction (0.0 to 1.0).
        probabilities: Probability distribution over all choices.
        raw_response: Raw response from Ollama API for debugging.
            May be a ChatResponse object (from score_fast) or a plain dict
            (from score_complete, which stores {"logprobs": ...}).
    """
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    raw_response: Any = field(default_factory=dict)


# Type alias for choices - can be a list of labels or dict mapping labels to descriptions
ChoicesType = Union[List[str], Dict[str, str]]