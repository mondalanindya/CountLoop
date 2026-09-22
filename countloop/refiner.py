"""Parameter-Free Textual Refinement Operator Psi for CountLoop.

Implements the textual analogue of backpropagation:
Translates Critic feedback P_feed into targeted, bounded edits on the
planning graph G = (V, E, B_bg), generating an explicit reasoning trace:
"Thought: ... Graph Edit: ..."
without modifying any model weights.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Set, Tuple

from countloop.config import CountLoopConfig
from countloop.types import (
    CriticFeedback,
    EditOperation,
    ObjectNode,
    PlanningGraph,
    SpatialRelation,
)


class TextualRefinementOperator:
    """Refinement operator Psi updating the planning graph using structured textual feedback."""

    def __init__(self, config: Optional[CountLoopConfig] = None):
        self.config = config or CountLoopConfig()
        self.rng = random.Random(self.config.seed)

    def refine(
        self,
        graph: PlanningGraph,
        feedback: CriticFeedback,
    ) -> Tuple[PlanningGraph, str, Set[str]]:
        """Applies Critic edits to the planning graph.

        Returns:
            Tuple of:
            - Updated PlanningGraph G'
            - Reasoning trace string ("Thought: ... Graph Edit: ...")
            - Set of edited instance IDs (used for incremental re-rendering cache)
        """
        edits = feedback.edits
        if not edits:
            return graph, "Thought: No edits requested by Critic. Graph remains unchanged.", set()

        thoughts: List[str] = []
        action_descriptions: List[str] = []
        edited_ids: Set[str] = set()

        # Clone current graph objects
        new_objects = [node.model_copy(deep=True) for node in graph.objects]
        new_relations = [rel.model_copy(deep=True) for rel in graph.relations]

        node_map = {node.id: node for node in new_objects}

        for edit in edits:
            if edit.type == "move":
                # Move overlapping instances away from each other
                if len(edit.targets) >= 2:
                    id1, id2 = edit.targets[0], edit.targets[1]
                    n1, n2 = node_map.get(id1), node_map.get(id2)
                    if n1 and n2:
                        dx = n1.x - n2.x
                        dy = n1.y - n2.y
                        dist = math.hypot(dx, dy)
                        if dist < 1e-4:
                            dx, dy = self.rng.uniform(-0.05, 0.05), self.rng.uniform(-0.05, 0.05)
                            dist = math.hypot(dx, dy)

                        # Bounded displacement <= max_displacement (0.08)
                        disp = min(self.config.max_displacement, 0.04)
                        shift_x = (dx / dist) * disp
                        shift_y = (dy / dist) * disp

                        n1.pos = [
                            max(0.05, min(0.95, n1.x + shift_x)),
                            max(0.05, min(0.95, n1.y + shift_y)),
                        ]
                        n2.pos = [
                            max(0.05, min(0.95, n2.x - shift_x)),
                            max(0.05, min(0.95, n2.y - shift_y)),
                        ]
                        edited_ids.add(n1.id)
                        edited_ids.add(n2.id)
                        thoughts.append(f"Separating overlapping pair {n1.id} and {n2.id} by delta ({shift_x:+.3f}, {shift_y:+.3f}).")
                        action_descriptions.append(f"Shift {n1.id} and {n2.id} to eliminate spatial overlap.")

            elif edit.type == "add":
                # Find available unoccupied space
                occupied = [(node.x, node.y) for node in new_objects]
                category = new_objects[0].category if new_objects else "object"
                # Determine how many to add from hint
                count_to_add = 1
                for token in edit.hint.split():
                    if token.isdigit():
                        count_to_add = min(int(token), 6)
                        break

                for _ in range(count_to_add):
                    new_pos = self._find_unoccupied_space(occupied)
                    occupied.append(new_pos)
                    new_id = f"{category}_{len(new_objects) + 1:02d}"

                    # Target area in [1/100, 1/25]
                    w = math.sqrt(self.config.min_bbox_area * 1.1)
                    h = math.sqrt(self.config.min_bbox_area * 1.1)
                    depth = float(max(0.1, min(0.9, 1.0 - new_pos[1])))

                    new_node = ObjectNode(
                        id=new_id,
                        category=category,
                        pos=[round(new_pos[0], 4), round(new_pos[1], 4)],
                        size=[round(w, 4), round(h, 4)],
                        depth=round(depth, 4),
                        color="natural",
                        attrs=["added in refinement"],
                    )
                    new_objects.append(new_node)
                    node_map[new_id] = new_node
                    edited_ids.add(new_id)

                thoughts.append(f"Critic detected count shortfall. Inserting {count_to_add} new node(s) in unoccupied regions.")
                action_descriptions.append(f"Insert {count_to_add} new {category} node(s) into planning graph.")

            elif edit.type == "remove":
                targets_to_remove = set(edit.targets)
                if not targets_to_remove and new_objects:
                    # Remove last object
                    targets_to_remove.add(new_objects[-1].id)

                remaining = []
                for node in new_objects:
                    if node.id in targets_to_remove:
                        edited_ids.add(node.id)
                    else:
                        remaining.append(node)
                new_objects = remaining
                node_map = {n.id: n for n in new_objects}
                thoughts.append(f"Critic detected excess count. Removing nodes: {list(targets_to_remove)}.")
                action_descriptions.append(f"Remove {len(targets_to_remove)} excess node(s).")

            elif edit.type == "resize":
                for target_id in edit.targets:
                    node = node_map.get(target_id)
                    if node:
                        # Scale up to satisfy minimum area (0.01)
                        if node.area < self.config.min_bbox_area:
                            scale = math.sqrt((self.config.min_bbox_area * 1.1) / node.area)
                            node.size = [
                                round(min(0.2, node.w * scale), 4),
                                round(min(0.2, node.h * scale), 4),
                            ]
                            edited_ids.add(node.id)
                thoughts.append(f"Resizing under-sized nodes {edit.targets} to satisfy visibility bound >= 1/100.")
                action_descriptions.append(f"Scale bounding boxes for {edit.targets}.")

            elif edit.type == "degrid":
                # Jitter all nodes slightly to eliminate linear alignment
                for node in new_objects:
                    jx = self.rng.uniform(-0.02, 0.02)
                    jy = self.rng.uniform(-0.02, 0.02)
                    node.pos = [
                        round(max(0.05, min(0.95, node.x + jx)), 4),
                        round(max(0.05, min(0.95, node.y + jy)), 4),
                    ]
                    edited_ids.add(node.id)
                thoughts.append("Degridding scene: injecting random coordinate jitter to break collinear rows and columns.")
                action_descriptions.append("Perturb all instance coordinates to restore natural spacing.")

        # Construct explicit reasoning trace
        thought_str = " ".join(thoughts)
        action_str = "; ".join(action_descriptions)
        reasoning_trace = f"Thought: {thought_str}\nGraph Edit: {action_str}"

        updated_graph = PlanningGraph(
            objects=new_objects,
            relations=new_relations,
            context=graph.context,
            prompts=graph.prompts,
        )

        return updated_graph, reasoning_trace, edited_ids

    def _find_unoccupied_space(
        self,
        occupied: List[Tuple[float, float]],
        max_attempts: int = 100,
    ) -> Tuple[float, float]:
        """Finds a candidate coordinate with maximum clearance from existing points."""
        best_pos = (0.5, 0.5)
        max_min_dist = -1.0

        for _ in range(max_attempts):
            x = self.rng.uniform(0.1, 0.9)
            y = self.rng.uniform(0.1, 0.9)
            if not occupied:
                return (x, y)

            min_d = min(math.hypot(x - ox, y - oy) for ox, oy in occupied)
            if min_d > max_min_dist:
                max_min_dist = min_d
                best_pos = (x, y)
                if min_d > 0.08:
                    break

        return best_pos
