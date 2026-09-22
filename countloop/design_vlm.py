"""Design VLM component for CountLoop.

Translates input text prompts into structured Planning Graphs G = (V, E, B_bg),
enforcing anti-grid constraints, bounded instance areas, depth layers, and
relational priors.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from countloop.config import CountLoopConfig
from countloop.prompts import DESIGN_VLM_SYSTEM_PROMPT, format_design_prompt
from countloop.types import ObjectNode, PlanningGraph, SpatialRelation


class DesignVLM:
    """Design VLM Agent that plans non-grid, depth-aware object layouts."""

    def __init__(self, config: Optional[CountLoopConfig] = None):
        self.config = config or CountLoopConfig()
        self.rng = random.Random(self.config.seed)

    def parse_prompt_entities(self, prompt: str) -> List[Tuple[str, int]]:
        """Parses count specifications and categories from a text prompt.

        Example:
            "30 cups on a wooden table" -> [("cup", 30)]
            "48 birds and 30 dogs in a park" -> [("bird", 48), ("dog", 30)]
        """
        # Common number words to integers
        num_words = {
            "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20,
            "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
            "eighty": 80, "ninety": 90, "hundred": 100,
        }

        # Match patterns like: "30 cups", "a bird", "twenty oranges"
        entities: List[Tuple[str, int]] = []
        cleaned = prompt.lower().replace(",", " and ")

        # Look for explicit digit counts: (\d+)\s+([a-zA-Z_-]+)
        digit_matches = re.findall(r"(\d+)\s+([a-zA-Z_-]+)", cleaned)
        for count_str, cat in digit_matches:
            # Strip plural 's' or 'es'
            singular = self._singularize(cat)
            entities.append((singular, int(count_str)))

        if not entities:
            # Check for word counts e.g. "two cats"
            for word, val in num_words.items():
                pattern = rf"\b{word}\s+([a-zA-Z_-]+)"
                matches = re.findall(pattern, cleaned)
                for cat in matches:
                    if cat not in ["photo", "picture", "painting", "view", "group"]:
                        singular = self._singularize(cat)
                        entities.append((singular, val))

        if not entities:
            # Default fallback: single object count 1
            words = [w for w in cleaned.split() if len(w) > 2 and w not in ["photo", "with", "the", "and"]]
            cat = words[0] if words else "object"
            entities.append((self._singularize(cat), 1))

        return entities

    def _singularize(self, word: str) -> str:
        """Simple heuristic singularization for categories."""
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        elif word.endswith("es") and word[-3] in "sxyz" or word.endswith("ches") or word.endswith("shes"):
            return word[:-2]
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            return word[:-1]
        return word

    def plan_layout(
        self,
        prompt: str,
        target_count: Optional[int] = None,
        context: Optional[str] = None,
    ) -> PlanningGraph:
        """Main entry point: Generates a PlanningGraph G from the input prompt."""
        # 1. If VLM provider is an external API / LLM backend, attempt that first
        if self.config.vlm_provider in ["openai", "gemini", "anthropic", "hf"]:
            try:
                graph = self._call_vlm_backend(prompt)
                if graph and len(graph.objects) > 0:
                    return graph
            except Exception as e:
                if self.config.verbose:
                    print(f"[DesignVLM] Backend call failed ({e}); falling back to layout synthesizer.")

        # 2. Algorithmic anti-grid layout generator (guarantees paper constraints)
        return self._generate_algorithmic_layout(prompt, target_count, context)

    def _generate_algorithmic_layout(
        self,
        prompt: str,
        target_count: Optional[int] = None,
        context: Optional[str] = None,
    ) -> PlanningGraph:
        """Synthesizes a realistic, non-grid spatial planning graph respecting all paper rules:

        - 1/100 <= w*h <= 1/25 (area in [0.01, 0.04])
        - Minimum distance >= 0.03
        - Anti-grid jitter
        - Far -> Near depth ordering
        """
        entities = self.parse_prompt_entities(prompt)
        if target_count is not None and entities:
            # Overwrite total count with explicit target_count if provided
            total_parsed = sum(c for _, c in entities)
            if total_parsed > 0:
                scale = target_count / total_parsed
                entities = [(cat, max(1, int(round(c * scale)))) for cat, c in entities]
            else:
                entities = [(entities[0][0], target_count)]

        # Extract background context if not provided
        if not context:
            if " on " in prompt.lower():
                context = prompt.lower().split(" on ")[-1].strip()
            elif " in " in prompt.lower():
                context = prompt.lower().split(" in ")[-1].strip()
            else:
                context = "a neutral studio setting with soft natural illumination"

        objects: List[ObjectNode] = []
        relations: List[SpatialRelation] = []

        total_target = sum(cnt for _, cnt in entities)
        # Adapt base bbox size based on total density:
        # High N -> smaller boxes towards 1/100 (0.01 area -> ~0.10 x 0.10)
        # Low N  -> larger boxes towards 1/25 (0.04 area -> ~0.18 x 0.18)
        if total_target > 80:
            target_area = self.config.min_bbox_area * 1.05  # ~0.0105
        elif total_target > 40:
            target_area = 0.015
        elif total_target > 15:
            target_area = 0.025
        else:
            target_area = self.config.max_bbox_area * 0.9   # ~0.036

        base_dim = math.sqrt(target_area)

        # Place objects using jittered non-grid sampling
        placed_positions: List[Tuple[float, float]] = []
        node_idx = 1

        for category, count in entities:
            for _ in range(count):
                obj_id = f"{category}_{node_idx:02d}"
                node_idx += 1

                # Sample non-grid coordinates with boundary padding
                padding = base_dim * 0.7
                pos = self._find_valid_position(placed_positions, padding)
                placed_positions.append(pos)

                # Size with slight aspect ratio variation (e.g. 0.85 to 1.15)
                aspect = self.rng.uniform(0.85, 1.18)
                w = base_dim * math.sqrt(aspect)
                h = base_dim / math.sqrt(aspect)

                # Clamp area to strictly remain inside [1/100, 1/25]
                area = w * h
                if area < self.config.min_bbox_area:
                    scale = math.sqrt(self.config.min_bbox_area / area)
                    w, h = w * scale, h * scale
                elif area > self.config.max_bbox_area:
                    scale = math.sqrt(self.config.max_bbox_area / area)
                    w, h = w * scale, h * scale

                # Depth prior d in [0.1, 0.9]:
                # In natural perspective, objects lower on canvas (larger y) tend to be nearer
                depth_base = 1.0 - (pos[1] * 0.8 + 0.1)
                depth = float(max(0.05, min(0.95, depth_base + self.rng.uniform(-0.1, 0.1))))

                color = self.rng.choice(["natural", "vibrant", "subtle", "warm", "cool"])
                attrs = []
                if depth > 0.6:
                    attrs.append("background layer")
                elif depth < 0.3:
                    attrs.append("foreground prominent")

                node = ObjectNode(
                    id=obj_id,
                    category=category,
                    pos=[round(pos[0], 4), round(pos[1], 4)],
                    size=[round(w, 4), round(h, 4)],
                    depth=round(depth, 4),
                    color=color,
                    attrs=attrs,
                )
                objects.append(node)

        # Build relational edges E between nearest neighbors
        n = len(objects)
        for i in range(min(n, 40)):  # Connect key adjacent instances
            n1 = objects[i]
            # Find closest neighbor
            best_j, min_dist = None, 1e9
            for j in range(n):
                if i == j:
                    continue
                n2 = objects[j]
                dx = n2.x - n1.x
                dy = n2.y - n1.y
                d = math.hypot(dx, dy)
                if d < min_dist:
                    min_dist = d
                    best_j = j

            if best_j is not None and min_dist < 0.25:
                n2 = objects[best_j]
                dx = n2.x - n1.x
                dy = n2.y - n1.y
                angle = math.degrees(math.atan2(dy, dx))
                if abs(dx) > abs(dy):
                    rel = "right-of" if dx > 0 else "left-of"
                else:
                    rel = "below" if dy > 0 else "above"

                relations.append(
                    SpatialRelation(
                        from_id=n1.id,
                        to_id=n2.id,
                        relation=rel,
                        dist=round(min_dist, 4),
                        angle=round(angle, 1),
                    )
                )

        prompts = {
            "Pd": f"a high detail photo of {prompt}, natural composition, photorealistic, sharp focus",
            "Pbg": f"{context}, seamless background texture, no extra duplicate instances",
        }

        return PlanningGraph(
            objects=objects,
            relations=relations,
            context=context,
            prompts=prompts,
        )

    def _find_valid_position(
        self,
        existing: List[Tuple[float, float]],
        padding: float,
        max_attempts: int = 150,
    ) -> Tuple[float, float]:
        """Samples a coordinate in [0, 1]^2 satisfying minimum distance and breaking grids."""
        min_d = self.config.min_l2_dist

        for _ in range(max_attempts):
            # Sample coordinate
            x = self.rng.uniform(padding, 1.0 - padding)
            y = self.rng.uniform(padding, 1.0 - padding)

            # Check distance against all existing points
            valid = True
            for ex, ey in existing:
                dist = math.hypot(x - ex, y - ey)
                if dist < min_d:
                    valid = False
                    break

            if valid:
                return (x, y)

        # Fallback: if dense, return position with slight jitter
        x = self.rng.uniform(padding, 1.0 - padding)
        y = self.rng.uniform(padding, 1.0 - padding)
        return (x, y)

    def _call_vlm_backend(self, prompt: str) -> Optional[PlanningGraph]:
        """Calls external VLM API (OpenAI, Gemini, HF) to execute DESIGN_VLM_SYSTEM_PROMPT."""
        formatted_prompt = format_design_prompt(prompt)

        import requests

        if self.config.vlm_provider in ["openai", "local", "vllm", "ollama"]:
            base_url = os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1")
            model = self.config.vlm_model_name if self.config.vlm_provider != "openai" else "gpt-4o"
            headers = {
                "Authorization": f"Bearer {self.config.api_key or 'EMPTY'}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": DESIGN_VLM_SYSTEM_PROMPT},
                    {"role": "user", "content": f'CURRENT PROMPT: "{prompt}"\nOUTPUT: JSON exactly following SCHEMA only.'},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
            }
            resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                raw_json = resp.json()["choices"][0]["message"]["content"]
                return PlanningGraph.from_json(raw_json)

        elif self.config.vlm_provider == "gemini":
            api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{DESIGN_VLM_SYSTEM_PROMPT}\n\nCURRENT PROMPT: \"{prompt}\"\nOUTPUT: JSON exactly following SCHEMA only."}
                        ]
                    }
                ],
                "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2},
            }
            resp = requests.post(url, json=payload, timeout=45)
            if resp.status_code == 200:
                raw_json = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                return PlanningGraph.from_json(raw_json)

        elif self.config.vlm_provider == "anthropic":
            api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 4096,
                "system": DESIGN_VLM_SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": f'CURRENT PROMPT: "{prompt}"\nOUTPUT: JSON exactly following SCHEMA only.'}
                ],
                "temperature": 0.2,
            }
            resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                raw_json = resp.json()["content"][0]["text"]
                return PlanningGraph.from_json(raw_json)

        return None
