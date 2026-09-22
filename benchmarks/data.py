"""Benchmark definitions for CountLoop-S and CountLoop-M.

As defined in Supplementary Section 4:
- CountLoop-S: Single-category, high instance counts (200 prompts, 30-200 objects)
- CountLoop-M: Multi-category, high instance counts (200 prompts, 30-200 objects)
Curated from 92 diverse categories across real-world contexts.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

# Full 92 categories curated from OmniCount-191 (Mondal et al., 2025)
OMNICOUNT_92_CATEGORIES: List[str] = [
    "airplane", "apple", "balloon", "banana", "bear", "bird", "boat", "book",
    "bottle", "bowl", "broccoli", "bus", "butterfly", "button", "cake", "candle",
    "can", "car", "carrot", "cat", "chair", "clock", "coin", "cookie", "cow",
    "cup", "deer", "dog", "donut", "duck", "elephant", "fish", "flower", "fork",
    "giraffe", "glass", "glove", "goat", "guitar", "hat", "helicopter", "horse",
    "hot air balloon", "kite", "knife", "lamp", "laptop", "lemon", "lion",
    "monkey", "motorcycle", "mouse", "mug", "orange", "panda", "peacock",
    "pen", "pencil", "penguin", "person", "pig", "pillow", "pineapple", "plate",
    "rabbit", "ring", "rose", "scissors", "sheep", "shoe", "spoon", "strawberry",
    "suitcase", "swan", "table", "teacup", "teddy bear", "tiger", "tomato",
    "toy", "train", "tree", "truck", "turtle", "umbrella", "vase", "violin",
    "watch", "whale", "wine glass", "zebra",
]

# Diverse real-world context templates
REAL_WORLD_CONTEXTS: List[str] = [
    "on a wooden table",
    "in a kitchen cabinet",
    "on a pantry shelf",
    "on a picnic blanket in the park",
    "in a marketplace display",
    "in the clear blue sky",
    "floating in calm water",
    "on a velvet cloth",
    "over a scenic green valley",
    "in an orchard bathed in sunlight",
    "on a jeweler presentation tray",
    "on a rustic cobblestone path",
]

# Curated benchmark sample prompts from CountLoop-S
COUNTLOOP_S_PROMPTS: List[Dict[str, Any]] = [
    {"prompt": "A photo of 30 cups on a wooden table", "category": "cup", "count": 30},
    {"prompt": "A photo of 45 apples in a market basket", "category": "apple", "count": 45},
    {"prompt": "A photo of 60 oranges in a wooden crate", "category": "orange", "count": 60},
    {"prompt": "A photo of 75 birds soaring in the sky", "category": "bird", "count": 75},
    {"prompt": "A photo of 90 buttons on a sewing table", "category": "button", "count": 90},
    {"prompt": "A photo of 120 balloons floating over a valley", "category": "balloon", "count": 120},
    {"prompt": "A photo of 140 oranges stacked on a display", "category": "orange", "count": 140},
    {"prompt": "A photo of 180 sheep grazing on a green hill", "category": "sheep", "count": 180},
    {"prompt": "A photo of 200 watches neatly arranged in a jeweler display", "category": "watch", "count": 200},
]

# Curated benchmark sample prompts from CountLoop-M
COUNTLOOP_M_PROMPTS: List[Dict[str, Any]] = [
    {"prompt": "A photo of 20 cats and 15 dogs in a sunny garden", "categories": ["cat", "dog"], "counts": [20, 15], "count": 35},
    {"prompt": "A photo of 48 birds and 30 dogs in a meadow", "categories": ["bird", "dog"], "counts": [48, 30], "count": 78},
    {"prompt": "A photo of 50 apples and 40 bananas on a kitchen island", "categories": ["apple", "banana"], "counts": [50, 40], "count": 90},
    {"prompt": "A photo of 60 balloons and 25 pineapples at a beach party", "categories": ["balloon", "pineapple"], "counts": [60, 25], "count": 85},
    {"prompt": "A photo of 140 oranges and 31 birds near an orchard", "categories": ["orange", "bird"], "counts": [140, 31], "count": 171},
]


def generate_countloop_s_dataset(n_prompts: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates the full 200 prompts for the CountLoop-S single-category benchmark."""
    rng = random.Random(seed)
    dataset: List[Dict[str, Any]] = []

    # Stratified count range: 30-60 (33%), 61-120 (33%), 121-200 (34%)
    for i in range(n_prompts):
        cat = OMNICOUNT_92_CATEGORIES[i % len(OMNICOUNT_92_CATEGORIES)]
        ctx = rng.choice(REAL_WORLD_CONTEXTS)

        # Distribute over tiers
        if i % 3 == 0:
            count = rng.randint(30, 60)
        elif i % 3 == 1:
            count = rng.randint(61, 120)
        else:
            count = rng.randint(121, 200)

        plural = f"{cat}s" if not cat.endswith("s") else cat
        prompt = f"A photo of {count} {plural} {ctx}"
        dataset.append({
            "id": f"countloop_s_{i+1:03d}",
            "prompt": prompt,
            "category": cat,
            "count": count,
            "tier": "30-60" if count <= 60 else ("61-120" if count <= 120 else "121-200"),
        })

    return dataset


def generate_countloop_m_dataset(n_prompts: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates the full 200 prompts for the CountLoop-M multi-category benchmark."""
    rng = random.Random(seed + 1)
    dataset: List[Dict[str, Any]] = []

    for i in range(n_prompts):
        cat1, cat2 = rng.sample(OMNICOUNT_92_CATEGORIES, 2)
        ctx = rng.choice(REAL_WORLD_CONTEXTS)

        total_count = rng.randint(30, 200)
        split = rng.uniform(0.25, 0.75)
        c1 = max(1, int(round(total_count * split)))
        c2 = max(1, total_count - c1)

        p1 = f"{cat1}s" if not cat1.endswith("s") else cat1
        p2 = f"{cat2}s" if not cat2.endswith("s") else cat2

        prompt = f"A photo of {c1} {p1} and {c2} {p2} {ctx}"
        dataset.append({
            "id": f"countloop_m_{i+1:03d}",
            "prompt": prompt,
            "categories": [cat1, cat2],
            "counts": [c1, c2],
            "count": c1 + c2,
        })

    return dataset
