"""Tests for CountLoop schemas, data structures, and types."""

import json
import pytest
from countloop.types import (
    CriticFeedback,
    CriticScores,
    EditOperation,
    ObjectNode,
    PlanningGraph,
    SpatialRelation,
)


def test_object_node_creation_and_bounds():
    node = ObjectNode(
        id="orange_01",
        category="orange",
        pos=[0.5, 0.5],
        size=[0.1, 0.1],
        depth=0.7,
        color="orange",
        attrs=["top layer"],
    )

    assert node.id == "orange_01"
    assert node.category == "orange"
    assert node.x == 0.5
    assert node.y == 0.5
    assert node.w == 0.1
    assert node.h == 0.1
    assert pytest.approx(node.area, 0.001) == 0.01

    bbox = node.bbox_xyxy
    assert pytest.approx(bbox[0], 0.001) == 0.45
    assert pytest.approx(bbox[1], 0.001) == 0.45
    assert pytest.approx(bbox[2], 0.001) == 0.55
    assert pytest.approx(bbox[3], 0.001) == 0.55

    pixel_box = node.pixel_bbox(1024, 1024)
    assert pixel_box == (461, 461, 563, 563)


def test_object_node_clamping():
    # Out-of-bounds pos and depth should be clamped to [0, 1]
    node = ObjectNode(
        id="clamped_node",
        category="cup",
        pos=[-0.2, 1.5],
        size=[0.1, 0.1],
        depth=-0.1,
    )
    assert node.pos == [0.0, 1.0]
    assert node.depth == 0.0


def test_planning_graph_serialization():
    node1 = ObjectNode(id="n1", category="cat", pos=[0.3, 0.4], size=[0.1, 0.1], depth=0.8)
    node2 = ObjectNode(id="n2", category="cat", pos=[0.6, 0.4], size=[0.1, 0.1], depth=0.2)
    rel = SpatialRelation(from_id="n1", to_id="n2", relation="left-of", dist=0.3, angle=0.0)

    graph = PlanningGraph(
        objects=[node1, node2],
        relations=[rel],
        context="wooden floor",
        prompts={"Pd": "two cats", "Pbg": "wooden floor"},
    )

    json_str = graph.to_json()
    loaded = PlanningGraph.from_json(json_str)

    assert loaded.num_instances == 2
    assert loaded.context == "wooden floor"
    assert loaded.relations[0].from_id == "n1"
    assert loaded.relations[0].to_id == "n2"
    assert loaded.relations[0].relation == "left-of"

    # Depth sorting Far -> Near: node1 (d=0.8) should come before node2 (d=0.2)
    sorted_nodes = loaded.depth_sorted_instances()
    assert sorted_nodes[0].id == "n1"
    assert sorted_nodes[1].id == "n2"


def test_critic_scores_and_feedback():
    scores = CriticScores(
        s_c=0.9,
        c_hat=27,
        N=30,
        s_a=0.85,
        C_acc=0.9,
        S=0.88,
        alpha=0.6,
    )

    feedback = CriticFeedback(
        scores=scores,
        decision={"continue": True, "reason": "S < tau"},
        feedback={
            "summary": "Looks good but short 3 cups",
            "count": "27 detected, target 30",
            "edits": [
                {"type": "add", "targets": [], "hint": "Add 3 cups in open spaces"},
                {"type": "move", "targets": ["cup_01", "cup_02"], "hint": "Separate overlapping pair"},
            ],
        },
    )

    assert len(feedback.edits) == 2
    assert feedback.edits[0].type == "add"
    assert feedback.edits[1].type == "move"
    assert feedback.edits[1].targets == ["cup_01", "cup_02"]
