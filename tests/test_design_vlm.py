"""Tests for Design VLM and layout planning."""

import pytest
from countloop.config import CountLoopConfig
from countloop.design_vlm import DesignVLM


def test_parse_prompt_entities_single():
    vlm = DesignVLM()
    entities = vlm.parse_prompt_entities("30 cups on a wooden table")
    assert len(entities) == 1
    assert entities[0][0] == "cup"
    assert entities[0][1] == 30


def test_parse_prompt_entities_multi():
    vlm = DesignVLM()
    entities = vlm.parse_prompt_entities("48 birds and 30 dogs in a meadow")
    categories = [cat for cat, _ in entities]
    counts = [c for _, c in entities]
    assert "bird" in categories
    assert "dog" in categories
    assert 48 in counts
    assert 30 in counts


def test_plan_layout_constraints():
    config = CountLoopConfig(seed=42)
    vlm = DesignVLM(config=config)

    graph = vlm.plan_layout("30 cups on a wooden table", target_count=30)
    assert graph.num_instances == 30

    # Test area bounds constraint: 1/100 <= w*h <= 1/25
    for node in graph.objects:
        assert node.area >= config.min_bbox_area - 1e-4, f"{node.id} area {node.area} < min"
        assert node.area <= config.max_bbox_area + 1e-4, f"{node.id} area {node.area} > max"
        assert 0.0 <= node.x <= 1.0
        assert 0.0 <= node.y <= 1.0
        assert 0.0 <= node.depth <= 1.0

    # Test that relations are generated
    assert len(graph.relations) > 0
    # Test prompts
    assert "cups" in graph.foreground_prompt or "photo" in graph.foreground_prompt
    assert "table" in graph.background_prompt


def test_plan_high_count():
    config = CountLoopConfig(seed=123)
    vlm = DesignVLM(config=config)

    # High instance count N=100
    graph = vlm.plan_layout("100 oranges in a crate", target_count=100)
    assert graph.num_instances == 100

    # Ensure all nodes meet paper constraints
    for node in graph.objects:
        assert node.area >= config.min_bbox_area - 1e-4
        assert node.area <= config.max_bbox_area + 1e-4
