"""Shared dataset evaluation helper for local_tests.

Contains the test dataset and a runner that classifies every entry with both
``classify()`` and ``generate()``, then writes results to a timestamped CSV.

A ``"none_of_the_above"`` choice is appended to the category list so the model
can express that no category fits -- its probability is reported in the
``prob_none`` column.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

from ollama_classifier import LLMClassifier

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

data_json = {
    "categories": [
        "technology",
        "food_cooking",
        "sports_fitness",
        "finance_investing",
        "travel_tourism",
    ],
    "dataset": [
        {
            "id": 1,
            "text": "The new smartphone features an upgraded octa-core processor and 12GB of RAM for seamless multitasking.",
            "ambiguity_level": "clear",
            "primary_category": "technology",
            "secondary_category": None,
        },
        {
            "id": 2,
            "text": "I need to update my operating system because some software applications are crashing on startup.",
            "ambiguity_level": "clear",
            "primary_category": "technology",
            "secondary_category": None,
        },
        {
            "id": 3,
            "text": "Cloud computing allows businesses to scale their server infrastructure dynamically based on demand.",
            "ambiguity_level": "clear",
            "primary_category": "technology",
            "secondary_category": None,
        },
        {
            "id": 4,
            "text": "Whisk the egg whites until stiff peaks form before gently folding them into the cake batter.",
            "ambiguity_level": "clear",
            "primary_category": "food_cooking",
            "secondary_category": None,
        },
        {
            "id": 5,
            "text": "This local Italian restaurant serves authentic wood-fired Neapolitan pizza with fresh basil and mozzarella.",
            "ambiguity_level": "clear",
            "primary_category": "food_cooking",
            "secondary_category": None,
        },
        {
            "id": 6,
            "text": "Slow-roasting garlic in olive oil at a low temperature produces a sweet, spreadable paste.",
            "ambiguity_level": "clear",
            "primary_category": "food_cooking",
            "secondary_category": None,
        },
        {
            "id": 7,
            "text": "The striker scored a stunning hat-trick in the second half to secure a victory for his team.",
            "ambiguity_level": "clear",
            "primary_category": "sports_fitness",
            "secondary_category": None,
        },
        {
            "id": 8,
            "text": "Proper hydration and stretching before running a marathon are essential to prevent muscle cramps.",
            "ambiguity_level": "clear",
            "primary_category": "sports_fitness",
            "secondary_category": None,
        },
        {
            "id": 9,
            "text": "Our local basketball league is looking for new referees to officiate the upcoming weekend games.",
            "ambiguity_level": "clear",
            "primary_category": "sports_fitness",
            "secondary_category": None,
        },
        {
            "id": 10,
            "text": "Diversifying your investment portfolio across stocks, bonds, and real estate helps mitigate risk.",
            "ambiguity_level": "clear",
            "primary_category": "finance_investing",
            "secondary_category": None,
        },
        {
            "id": 11,
            "text": "The central bank decided to raise interest rates to curb rising inflation across the country.",
            "ambiguity_level": "clear",
            "primary_category": "finance_investing",
            "secondary_category": None,
        },
        {
            "id": 12,
            "text": "Opening a high-yield savings account is a simple way to earn interest on your emergency fund.",
            "ambiguity_level": "clear",
            "primary_category": "finance_investing",
            "secondary_category": None,
        },
        {
            "id": 13,
            "text": "We spent the afternoon exploring the historic ruins of Rome and taking photos of the Colosseum.",
            "ambiguity_level": "clear",
            "primary_category": "travel_tourism",
            "secondary_category": None,
        },
        {
            "id": 14,
            "text": "Remember to check the visa requirements and passport validity before booking your flights abroad.",
            "ambiguity_level": "clear",
            "primary_category": "travel_tourism",
            "secondary_category": None,
        },
        {
            "id": 15,
            "text": "The boutique hotel offers stunning ocean views and is located just steps from the sandy beach.",
            "ambiguity_level": "clear",
            "primary_category": "travel_tourism",
            "secondary_category": None,
        },
        {
            "id": 16,
            "text": "Backpacking through Southeast Asia is an affordable way for students to experience diverse cultures.",
            "ambiguity_level": "clear",
            "primary_category": "travel_tourism",
            "secondary_category": None,
        },
        {
            "id": 17,
            "text": "This new smart air fryer connects to your home Wi-Fi, allowing you to monitor cooking progress from an app.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "technology",
            "secondary_category": "food_cooking",
        },
        {
            "id": 18,
            "text": "I bought a premium smartwatch to track my daily steps, heart rate variability, and GPS routes during morning jogs.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "technology",
            "secondary_category": "sports_fitness",
        },
        {
            "id": 19,
            "text": "While wandering the streets of Paris, I stumbled upon a tiny bakery serving the most incredible butter croissants.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "food_cooking",
            "secondary_category": "travel_tourism",
        },
        {
            "id": 20,
            "text": "I used a digital kitchen scale and a specialized molecular gastronomy calculator to measure the sodium alginate for this recipe.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "food_cooking",
            "secondary_category": "technology",
        },
        {
            "id": 21,
            "text": "The professional football player signed a multi-million dollar contract extension, making him the highest-paid athlete this season.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "sports_fitness",
            "secondary_category": "finance_investing",
        },
        {
            "id": 22,
            "text": "The cycling team utilized wind-tunnel data and advanced computational fluid dynamics software to optimize their riding postures.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "sports_fitness",
            "secondary_category": "technology",
        },
        {
            "id": 23,
            "text": "The sudden surge in cryptocurrency trading caused several online brokerage platforms to experience temporary server outages.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "finance_investing",
            "secondary_category": "technology",
        },
        {
            "id": 24,
            "text": "Many digital nomads set up offshore bank accounts to optimize their tax liabilities while moving between different countries.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "finance_investing",
            "secondary_category": "travel_tourism",
        },
        {
            "id": 25,
            "text": "Budgeting for a year-long trip around the world requires calculating daily accommodation costs and saving thousands in advance.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "travel_tourism",
            "secondary_category": "finance_investing",
        },
        {
            "id": 26,
            "text": "The culinary tourism package includes guided street food tours and private cooking classes with local chefs in Tokyo.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "travel_tourism",
            "secondary_category": "food_cooking",
        },
        {
            "id": 27,
            "text": "Rising grain prices and supply chain disruptions are forcing local artisan bakeries to increase the cost of a sourdough loaf.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "food_cooking",
            "secondary_category": "finance_investing",
        },
        {
            "id": 28,
            "text": "The software company introduced a new subscription model for its cloud services, aiming to boost recurring software-as-a-service revenues.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "technology",
            "secondary_category": "finance_investing",
        },
        {
            "id": 29,
            "text": "Our amateur soccer club is traveling to Spain next month to participate in an international friendly tournament.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "sports_fitness",
            "secondary_category": "travel_tourism",
        },
        {
            "id": 30,
            "text": "Investing in premium sports memorabilia, like game-worn jerseys, has become a highly lucrative alternative asset class.",
            "ambiguity_level": "mildly_ambiguous",
            "primary_category": "finance_investing",
            "secondary_category": "sports_fitness",
        },
        {
            "id": 31,
            "text": "This article reviews the engineering behind elite running shoes, comparing the energy-return polymer plates with smart embedded pressure sensors.",
            "ambiguity_level": "highly_ambiguous",
            "primary_category": "technology",
            "secondary_category": "sports_fitness",
        },
        {
            "id": 32,
            "text": "Mobile banking apps are leveraging decentralized blockchain protocols and biometric authentication to secure financial transactions.",
            "ambiguity_level": "highly_ambiguous",
            "primary_category": "technology",
            "secondary_category": "finance_investing",
        },
        {
            "id": 33,
            "text": "A comprehensive guide to exploring the night markets of Taiwan, focusing on the history of regional street food and how to navigate the crowded stalls.",
            "ambiguity_level": "highly_ambiguous",
            "primary_category": "food_cooking",
            "secondary_category": "travel_tourism",
        },
        {
            "id": 34,
            "text": "An analysis of the global coffee bean futures market, discussing how climate change impacts crop yields and the final retail price of espresso.",
            "ambiguity_level": "highly_ambiguous",
            "primary_category": "food_cooking",
            "secondary_category": "finance_investing",
        },
        {
            "id": 35,
            "text": "Hiking the Pacific Crest Trail: A detailed breakdown of the physical conditioning required for high-altitude trekking and the logistics of navigating national parks.",
            "ambiguity_level": "highly_ambiguous",
            "primary_category": "sports_fitness",
            "secondary_category": "travel_tourism",
        },
        {
            "id": 36,
            "text": "A sports nutritionist's guide to meal prepping, detailing exactly what macro-nutrients to eat before high-intensity interval training to maximize muscle recovery.",
            "ambiguity_level": "highly_ambiguous",
            "primary_category": "sports_fitness",
            "secondary_category": "food_cooking",
        },
        {
            "id": 37,
            "text": "Agriculture technology is evolving rapidly, with automated indoor hydroponic systems using AI sensors to deliver nutrients to crops without soil.",
            "ambiguity_level": "highly_ambiguous",
            "primary_category": "technology",
            "secondary_category": "food_cooking",
        },
        {
            "id": 38,
            "text": "Analyzing the economic impact of international tourism on developing island nations, specifically tracking foreign currency exchange and hotel industry revenues.",
            "ambiguity_level": "highly_ambiguous",
            "primary_category": "finance_investing",
            "secondary_category": "travel_tourism",
        },
        {
            "id": 39,
            "text": "The chemical structure of DNA consists of two long chains of nucleotides twisted into a double helix.",
            "ambiguity_level": "out_of_scope",
            "primary_category": None,
            "secondary_category": None,
        },
        {
            "id": 40,
            "text": "William Shakespeare's tragedy 'Hamlet' explores themes of revenge, madness, and moral corruption in the Danish court.",
            "ambiguity_level": "out_of_scope",
            "primary_category": None,
            "secondary_category": None,
        },
    ],
}

# The "none of the above" choice appended so the model can reject all categories
NONE_CHOICE = "none_of_the_above"


def run_dataset_and_save_csv(
    classifier: LLMClassifier,
    backend_name: str,
    llm_name: str,
) -> str:
    """Classify every dataset entry with both APIs and save to CSV.

    A ``"none_of_the_above"`` choice is added to the category list so the
    model can signal that none of the real categories fit. Its probability
    is reported in the ``prob_none`` column.

    Args:
        classifier: An ``LLMClassifier`` instance.
        backend_name: Short backend name (e.g. ``"ollama"``).
        llm_name: Model name (e.g. ``"qwen2.5:3b-instruct"``).

    Returns:
        The path to the generated CSV file.
    """
    categories = data_json["categories"]
    entries = data_json["dataset"]

    # Choices presented to the classifier: real categories + none_of_the_above
    choices = categories + [NONE_CHOICE]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_path = os.path.join(os.path.dirname(__file__), f"{backend_name}_{timestamp}.csv")

    fieldnames = (
        ["id", "text", "ambiguity_level", "primary_category", "secondary_category",
         "backend", "llm", "api", "prediction", "confidence"]
        + [f"prob_{c}" for c in categories]
        + ["prob_none"]
    )

    rows: list[dict[str, object]] = []

    for entry in entries:
        text = entry["text"]

        for api_name in ("classify", "generate"):
            if api_name == "classify":
                result = classifier.classify(text=text, choices=choices)
            else:
                result = classifier.generate(text=text, choices=choices, max_calls=1)

            row: dict[str, object] = {
                "id": entry["id"],
                "text": text,
                "ambiguity_level": entry["ambiguity_level"],
                "primary_category": entry["primary_category"] or "",
                "secondary_category": entry["secondary_category"] or "",
                "backend": backend_name,
                "llm": llm_name,
                "api": api_name,
                "prediction": result.prediction,
                "confidence": f"{result.confidence:.6f}",
            }
            for cat in categories:
                row[f"prob_{cat}"] = f"{result.probabilities.get(cat, 0.0):.6f}"
            row["prob_none"] = f"{result.probabilities.get(NONE_CHOICE, 0.0):.6f}"
            rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  CSV saved: {csv_path} ({len(rows)} rows)")
    return csv_path
