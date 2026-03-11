"""Prompt building utilities for classification."""

from typing import List, Dict, Union


def build_classification_prompt(
    text: str,
    choices: Union[List[str], Dict[str, str]],
    system_prompt: str | None = None,
) -> tuple[str, str]:
    """Build the system and user prompts for classification.
    
    Args:
        text: The text to classify.
        choices: Either a list of choice labels, or a dict mapping labels to descriptions.
        system_prompt: Optional custom system prompt.
        
    Returns:
        Tuple of (system_prompt, user_prompt) strings.
    """
    # Build the choices section of the prompt
    choices_text = _format_choices(choices)
    
    # Default system prompt
    if system_prompt is None:
        system_prompt = (
            "You are a precise text classifier. "
            "Your task is to classify the given text into exactly one of the provided categories. "
            "Respond with only the category label, nothing else."
        )
    
    # Build user prompt
    user_prompt = f"""Classify the following text into one of these categories:

{choices_text}

Text to classify:
{text}

Respond with only the category label."""
    
    return system_prompt, user_prompt


def _format_choices(choices: Union[List[str], Dict[str, str]]) -> str:
    """Format choices for the prompt.
    
    Args:
        choices: Either a list of choice labels, or a dict mapping labels to descriptions.
        
    Returns:
        Formatted string representation of choices.
    """
    if isinstance(choices, dict):
        # Choices with descriptions
        lines = []
        for label, description in choices.items():
            lines.append(f"- {label}: {description}")
        return "\n".join(lines)
    else:
        # Simple list of choices
        return "\n".join(f"- {choice}" for choice in choices)


def get_choice_labels(choices: Union[List[str], Dict[str, str]]) -> List[str]:
    """Extract the choice labels from either format.
    
    Args:
        choices: Either a list of choice labels, or a dict mapping labels to descriptions.
        
    Returns:
        List of choice labels.
    """
    if isinstance(choices, dict):
        return list(choices.keys())
    return list(choices)


def build_json_schema_for_choices(choices: List[str]) -> dict:
    """Build a JSON schema that constrains output to the given choices.
    
    Args:
        choices: List of valid choice labels.
        
    Returns:
        JSON schema dict for structured output.
    """
    return {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": choices,
            }
        },
        "required": ["label"],
    }