"""Tests for Critic VLM and Parameter-Free Textual Refinement Operator Psi."""

import pytest
from PIL import Image

from countloop.config import CountLoopConfig
from countloop.critic import CriticVLM
from countloop.refiner import TextualRefinementOperator
from countloop.types import (
    CriticFeedback,
    CriticScores,
    EditOperation,
    ObjectNode,
    PlanningGraph,
)


def test_critic_scoring_formulas():
    config = CountLoopConfig(alpha=0.6, tau=0.85)
    critic = CriticVLM(config=config)

    # Synthetic image & graph with 25 objects against target 30
    nodes = [
        ObjectNode(id=f"cup_{i:02d}", category="cup", pos=[0.1 + (i % 5) * 0.15, 0.1 + (i // 5) * 0.15], size=[0.08, 0.08])
        for i in range(25)
    ]
    graph = PlanningGraph(objects=nodes)
    img = Image.new("RGB", (512, 512), color=(255, 255, 255))

    scores, feedback = critic.evaluate(img, graph, target_count=30)

    # c_hat = 25, c_gt = 30 -> error = 5 -> s_c = 1 - 5/30 = 25/30 = 0.8333
    assert scores.detected_count == 25
    assert scores.target_count == 30
    assert pytest.approx(scores.s_c, 0.01) == 0.8333

    # Composite S = 0.6 * s_c + 0.4 * s_a
    expected_s = 0.6 * scores.s_c + 0.4 * scores.s_a
    assert pytest.approx(scores.S, 0.001) == expected_s

    # Advisory decision should request continue because S < 0.85
    assert feedback.decision["continue"] == (scores.S < 0.85)


def test_refiner_psi_add_operation():
    config = CountLoopConfig()
    refiner = TextualRefinementOperator(config=config)

    node = ObjectNode(id="cup_01", category="cup", pos=[0.3, 0.3], size=[0.1, 0.1])
    graph = PlanningGraph(objects=[node])

    feedback = CriticFeedback(
        scores=CriticScores(s_c=0.5, c_hat=1, N=2, s_a=0.9, S=0.66),
        decision={"continue": True, "reason": "need more"},
        feedback={
            "summary": "1 cup present, need 1 more",
            "count": "short 1",
            "edits": [EditOperation(type="add", targets=[], hint="Add 1 cup in free region")],
        },
    )

    updated_graph, trace, edited_ids = refiner.refine(graph, feedback)

    assert updated_graph.num_instances == 2
    assert "Thought:" in trace
    assert "Graph Edit:" in trace
    assert len(edited_ids) > 0


def test_refiner_psi_move_bounded_displacement():
    config = CountLoopConfig(max_displacement=0.08)
    refiner = TextualRefinementOperator(config=config)

    # Overlapping pair
    n1 = ObjectNode(id="cup_01", category="cup", pos=[0.5, 0.5], size=[0.1, 0.1])
    n2 = ObjectNode(id="cup_02", category="cup", pos=[0.51, 0.51], size=[0.1, 0.1])
    graph = PlanningGraph(objects=[n1, n2])

    feedback = CriticFeedback(
        scores=CriticScores(s_c=1.0, c_hat=2, N=2, s_a=0.6, S=0.84),
        decision={"continue": True, "reason": "overlap"},
        feedback={
            "summary": "Cups overlap",
            "count": "2 detected",
            "edits": [EditOperation(type="move", targets=["cup_01", "cup_02"], hint="Increase separation")],
        },
    )

    updated_graph, trace, edited_ids = refiner.refine(graph, feedback)

    u1 = updated_graph.get_node("cup_01")
    u2 = updated_graph.get_node("cup_02")

    # Shift displacement must be <= 0.08
    d1 = ((u1.x - 0.5)**2 + (u1.y - 0.5)**2)**0.5
    d2 = ((u2.x - 0.51)**2 + (u2.y - 0.51)**2)**0.5
    assert d1 <= 0.08 + 1e-4
    assert d2 <= 0.08 + 1e-4

    # Distance between them must have increased
    new_dist = ((u1.x - u2.x)**2 + (u1.y - u2.y)**2)**0.5
    old_dist = ((0.5 - 0.51)**2 + (0.5 - 0.51)**2)**0.5
    assert new_dist > old_dist
