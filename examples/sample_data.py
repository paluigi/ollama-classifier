"""Sample datasets for testing ollama-classifier.

Provides two ready-to-use datasets for text classification:

- ``DATASET_WITHOUT_DESCRIPTIONS``: labels as a plain list.
- ``DATASET_WITH_DESCRIPTIONS``: labels as a dict with descriptions.

Both datasets contain the same 20 short texts representing customer support
tickets across four categories.  Each text includes an ``expected_label``
so you can verify the classifier's accuracy.

Quick start::

    from examples.sample_data import DATASET_WITHOUT_DESCRIPTIONS

    results = classifier.batch_generate(
        texts=DATASET_WITHOUT_DESCRIPTIONS.texts,
        choices=DATASET_WITHOUT_DESCRIPTIONS.choices,
    )
"""

from dataclasses import dataclass, field
from typing import List, Dict

# ── Raw data ────────────────────────────────────────────────────────────────

TEXTS: List[str] = [
    # billing (5)
    "I was charged twice for my last order",
    "Can I get a refund for the subscription I cancelled last week?",
    "My invoice shows a different amount than what was quoted",
    "Where can I find my payment history?",
    "I need an update on my pending refund",
    # technical_support (5)
    "The app keeps crashing when I try to upload a file",
    "I can't log in to my account after the latest update",
    "The page loads very slowly on mobile devices",
    "How do I reset my password if I don't have access to my email?",
    "I'm getting a 404 error on the dashboard",
    # account (5)
    "How do I change the email address on my profile?",
    "I'd like to delete my account and all associated data",
    "Can I upgrade from the free plan to the premium plan?",
    "How do I add a second user to my team account?",
    "I need to update my billing address",
    # general (5)
    "What are your business hours?",
    "Is there a mobile app available?",
    "Do you offer discounts for non-profit organizations?",
    "Where can I find your privacy policy?",
    "How do I contact your customer support team?",
]

LABELS: List[str] = [
    "billing",
    "technical_support",
    "account",
    "general",
]

LABELS_WITH_DESCRIPTIONS: Dict[str, str] = {
    "billing": "Questions about charges, invoices, payments, refunds, and subscription costs",
    "technical_support": "Issues with software, bugs, errors, login problems, or performance",
    "account": "Requests to manage profile settings, plans, team members, or data",
    "general": "General inquiries about the company, policies, hours, or availability",
}

EXPECTED_LABELS: List[str] = [
    # billing
    "billing",
    "billing",
    "billing",
    "billing",
    "billing",
    # technical_support
    "technical_support",
    "technical_support",
    "technical_support",
    "technical_support",
    "technical_support",
    # account
    "account",
    "account",
    "account",
    "account",
    "account",
    # general
    "general",
    "general",
    "general",
    "general",
    "general",
]


# ── Dataset containers ─────────────────────────────────────────────────────


@dataclass
class SampleDataset:
    """A self-contained dataset ready to pass into the classifier.

    Attributes:
        texts: Short texts to classify.
        choices: Either a ``list`` of labels or a ``dict`` mapping labels to
                 descriptions.
        expected_labels: Expected correct label for each text (same order as
                         *texts*).
        description: Human-readable description of the dataset.
    """

    texts: List[str] = field(default_factory=list)
    choices: list | dict = field(default_factory=list)
    expected_labels: List[str] = field(default_factory=list)
    description: str = ""


# ── Public datasets ─────────────────────────────────────────────────────────

#: Dataset with labels only (plain list).
DATASET_WITHOUT_DESCRIPTIONS = SampleDataset(
    texts=TEXTS,
    choices=LABELS,
    expected_labels=EXPECTED_LABELS,
    description="20 customer support tickets classified into 4 categories "
                "(billing, technical_support, account, general) without "
                "label descriptions.",
)

#: Dataset with labels enriched with descriptions.
DATASET_WITH_DESCRIPTIONS = SampleDataset(
    texts=TEXTS,
    choices=LABELS_WITH_DESCRIPTIONS,
    expected_labels=EXPECTED_LABELS,
    description="20 customer support tickets classified into 4 categories "
                "(billing, technical_support, account, general) with "
                "label descriptions for improved accuracy.",
)
