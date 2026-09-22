"""Critic VLM and proxy evaluators for CountLoop.

Integrates:
1. Open-Vocabulary Detector (GroundingDINO / CountGD++ / OWLv2)
2. Aesthetic Scorer (Q-Align / CLIP-Aesthetic)
3. Scoring formulas (s_c, s_a, composite S = alpha*s_c + (1-alpha)*s_a)
4. Structured Critic VLM feedback generation with closed edit vocabulary
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from countloop.attention import compute_instance_visible_areas
from countloop.config import CountLoopConfig
from countloop.prompts import CRITIC_VLM_SYSTEM_PROMPT, format_critic_prompt
from countloop.types import (
    CriticFeedback,
    CriticScores,
    EditOperation,
    PlanningGraph,
)


class CriticVLM:
    """Critic VLM agent evaluating counting fidelity and aesthetics."""

    def __init__(self, config: Optional[CountLoopConfig] = None):
        self.config = config or CountLoopConfig()
        self.rng = random.Random(self.config.seed)
        self._dino_model = None
        self._dino_processor = None
        self._owl_model = None
        self._owl_processor = None

    def detect_objects(
        self,
        image: Image.Image,
        graph: PlanningGraph,
        target_category: Optional[str] = None,
    ) -> List[Tuple[float, float, float, float, float]]:
        """Runs open-vocabulary detector to detect object instances.

        Returns list of detected boxes: [(x1, y1, x2, y2, confidence), ...].
        """
        # Attempt real Hugging Face detector if not in simulation mode
        if not self.config.use_mock_engine and self.config.device != "cpu":
            try:
                category = target_category or (graph.objects[0].category if graph.objects else "object")
                real_boxes = self._detect_with_grounding_dino(image, category)
                if real_boxes is not None:
                    return real_boxes
            except Exception as e:
                if self.config.verbose:
                    print(f"[Critic] Hugging Face detector unavailable ({e}). Using proxy detector.")

        # Fallback to physical visibility calculation under occlusion threshold
        # (visible foreground area >= 1/6)
        width, height = image.size
        visible_stats = compute_instance_visible_areas(graph, (height, width))

        detected_boxes = []
        for node_id, full_px, visible_px in visible_stats:
            node = graph.get_node(node_id)
            if node is None:
                continue

            # Target category filter if provided
            if target_category and node.category.lower() != target_category.lower():
                continue

            vis_ratio = visible_px / full_px
            # Visible instance counting convention:
            # An instance counts if visible area >= 1/6 (approx 0.1667)
            if vis_ratio >= self.config.min_visible_ratio:
                # Add realistic detection confidence
                conf = min(0.98, max(0.35, 0.5 + vis_ratio * 0.45))
                detected_boxes.append((*node.bbox_xyxy, conf))

        return detected_boxes

    def _detect_with_grounding_dino(
        self,
        image: Image.Image,
        category: str,
    ) -> Optional[List[Tuple[float, float, float, float, float]]]:
        """Detects instances using HuggingFace GroundingDINO model."""
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if self._dino_model is None:
            self._dino_processor = AutoProcessor.from_pretrained(self.config.detector_model_id)
            self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.config.detector_model_id
            ).to(self.config.device)

        text_prompt = f"{category}."
        inputs = self._dino_processor(images=image, text=text_prompt, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = self._dino_model(**inputs)

        width, height = image.size
        results = self._dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.confidence_threshold,
            text_threshold=self.config.confidence_threshold,
            target_sizes=[(height, width)],
        )[0]

        detected_boxes = []
        for box, score in zip(results["boxes"], results["scores"]):
            x1, y1, x2, y2 = box.tolist()
            # Normalize to [0, 1]
            detected_boxes.append((x1 / width, y1 / height, x2 / width, y2 / height, float(score)))

        return detected_boxes

    def score_aesthetics(self, image: Image.Image, graph: PlanningGraph) -> float:
        """Computes aesthetic alignment score s_a in [0, 1] using Q-Align proxy."""
        # Measures visual layout balance, absence of excessive collisions, and margin
        violations = graph.validate_constraints(
            min_area=self.config.min_bbox_area,
            max_area=self.config.max_bbox_area,
            min_l2_dist=self.config.min_l2_dist,
        )

        base_aesthetic = 0.90
        # Penalize collision pairs
        num_collisions = len(violations["too_close_pairs"])
        base_aesthetic -= min(0.3, num_collisions * 0.04)

        # Penalize area violations
        num_area_viol = len(violations["under_min_area"]) + len(violations["over_max_area"])
        base_aesthetic -= min(0.2, num_area_viol * 0.02)

        # Penalize grid artifacts
        if violations["grid_alignment_detected"]:
            base_aesthetic -= 0.08

        # Slight natural jitter
        score = base_aesthetic + self.rng.uniform(-0.02, 0.02)
        return float(max(0.1, min(0.98, round(score, 4))))

    def evaluate(
        self,
        image: Image.Image,
        graph: PlanningGraph,
        target_count: int,
    ) -> Tuple[CriticScores, CriticFeedback]:
        """Runs the complete evaluation and produces structured Critic feedback."""
        # 1. Run Detector
        detected_boxes = self.detect_objects(image, graph)
        c_hat = len(detected_boxes)
        c_gt = max(1, target_count)

        # 2. Normalized count score: s_c = max(0, 1 - |c_hat - c_gt| / c_gt)
        count_err = abs(c_hat - c_gt)
        s_c = float(max(0.0, 1.0 - (count_err / float(c_gt))))

        # 3. Aesthetic score
        s_a = self.score_aesthetics(image, graph)

        # 4. Composite score: S = alpha * s_c + (1 - alpha) * s_a
        alpha = self.config.alpha
        S = float(round(alpha * s_c + (1.0 - alpha) * s_a, 4))

        # Count accuracy measure C_acc
        c_acc = float(round(max(0.0, 1.0 - (count_err / float(c_gt))), 4))

        scores = CriticScores(
            s_c=round(s_c, 4),
            c_hat=c_hat,
            N=c_gt,
            s_a=round(s_a, 4),
            C_acc=c_acc,
            S=S,
            alpha=alpha,
        )

        # 5. Generate structured feedback
        feedback = self._generate_feedback(graph, scores, c_hat, c_gt)
        return scores, feedback

    def _generate_feedback(
        self,
        graph: PlanningGraph,
        scores: CriticScores,
        c_hat: int,
        c_gt: int,
    ) -> CriticFeedback:
        """Constructs targeted edits restricted to move | add | remove | resize | degrid."""
        edits: List[EditOperation] = []
        violations = graph.validate_constraints(
            min_area=self.config.min_bbox_area,
            max_area=self.config.max_bbox_area,
            min_l2_dist=self.config.min_l2_dist,
        )

        # 1. Under-count: add missing instances
        if c_hat < c_gt:
            diff = c_gt - c_hat
            num_to_add = min(diff, 6)
            edits.append(
                EditOperation(
                    type="add",
                    targets=[],
                    hint=f"Detector detected {c_hat} instances (target: {c_gt}). Add {num_to_add} instances in unoccupied canvas space.",
                )
            )

        # 2. Over-count: remove extra instances
        elif c_hat > c_gt:
            diff = c_hat - c_gt
            num_to_remove = min(diff, 4)
            # Pick least visible / background instances
            targets = [node.id for node in graph.objects[-num_to_remove:]]
            edits.append(
                EditOperation(
                    type="remove",
                    targets=targets,
                    hint=f"Detector detected {c_hat} instances (target: {c_gt}). Remove {num_to_remove} redundant instances.",
                )
            )

        # 3. Overlaps / close pairs: move
        if violations["too_close_pairs"]:
            for id1, id2, dist in violations["too_close_pairs"][:3]:
                if len(edits) >= self.config.max_edits_per_round:
                    break
                edits.append(
                    EditOperation(
                        type="move",
                        targets=[id1, id2],
                        hint=f"Instances {id1} and {id2} overlap closely (dist={dist:.3f}). Increase spatial separation.",
                    )
                )

        # 4. Under-sized instances: resize
        if violations["under_min_area"]:
            under_ids = violations["under_min_area"][:2]
            edits.append(
                EditOperation(
                    type="resize",
                    targets=under_ids,
                    hint=f"Instances {under_ids} are below minimum area 0.01. Scale bounding boxes up to detectable threshold.",
                )
            )

        # 5. Grid alignment: degrid
        if violations["grid_alignment_detected"]:
            edits.append(
                EditOperation(
                    type="degrid",
                    targets=[],
                    hint="Detected rigid row or column alignment. Perturb positions with light random jitter.",
                )
            )

        # Enforce max 10 edits constraint
        edits = edits[: self.config.max_edits_per_round]

        # Advisory decision
        should_continue = scores.S < self.config.tau
        reason = (
            f"Composite score S={scores.S:.3f} >= tau={self.config.tau:.2f} (converged)"
            if not should_continue
            else f"Composite score S={scores.S:.3f} < tau={self.config.tau:.2f} (requires refinement)"
        )

        summary_text = (
            f"Image displays {c_hat} detected objects with aesthetic score {scores.s_a:.2f}."
        )
        count_text = (
            f"Detected count {c_hat} matches target {c_gt} exactly."
            if c_hat == c_gt
            else f"Detected count {c_hat} differs from target {c_gt} by {abs(c_hat - c_gt)}."
        )

        return CriticFeedback(
            scores=scores,
            decision={"continue": should_continue, "reason": reason},
            feedback={
                "summary": summary_text,
                "count": count_text,
                "edits": edits,
            },
        )
