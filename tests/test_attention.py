"""Tests for layout-aligned attention masking and cumulative composition."""

import numpy as np
import pytest
from countloop.attention import (
    apply_attention_masking,
    check_occlusion_constraints,
    compute_instance_visible_areas,
    create_bbox_mask,
    cumulative_latent_composition,
    self_segmentation_refinement,
)
from countloop.types import ObjectNode, PlanningGraph


def test_create_bbox_mask():
    # Box covering center 50%
    mask = create_bbox_mask((0.25, 0.25, 0.75, 0.75), 100, 100)
    assert mask.shape == (100, 100)
    assert mask[0, 0] == 0.0
    assert mask[99, 99] == 0.0
    assert mask[50, 50] == 1.0
    # Expected area = 50 * 50 = 2500
    assert mask.sum() == 2500


def test_self_segmentation_refinement():
    mask = create_bbox_mask((0.2, 0.2, 0.8, 0.8), 50, 50)
    refined = self_segmentation_refinement(mask)
    assert refined.shape == mask.shape
    # Refined ellipse has fewer pixels than square bounding box
    assert 0 < refined.sum() <= mask.sum()


def test_attention_masking_equation():
    # A_mask = A_cross * M_hat
    cross_attn = np.ones((32, 32, 4), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    a_mask = apply_attention_masking(cross_attn, mask)
    assert a_mask.shape == (32, 32, 4)
    # Outside mask must be zero
    assert np.all(a_mask[0:8, 0:8] == 0.0)
    # Inside mask must be 1.0
    assert np.all(a_mask[12:20, 12:20] == 1.0)


def test_cumulative_latent_composition_math():
    # F_{i+1} = 1_{l_i} * A_mask + (1 - 1_{l_i}) * F_i
    f_prev = np.full((32, 32, 4), 2.0, dtype=np.float32)
    a_mask = np.full((32, 32, 4), 5.0, dtype=np.float32)

    ind = np.zeros((32, 32), dtype=np.float32)
    ind[10:20, 10:20] = 1.0

    f_next = cumulative_latent_composition(f_prev, a_mask, ind)
    assert f_next.shape == (32, 32, 4)
    # Inside box: replaced with a_mask (5.0)
    assert np.all(f_next[12:18, 12:18] == 5.0)
    # Outside box: retained f_prev (2.0)
    assert np.all(f_next[0:5, 0:5] == 2.0)


def test_instance_visibility_and_occlusion():
    # Node 1 is Far (depth 0.8), Node 2 is Near (depth 0.2) and overlaps 50% of Node 1
    node1 = ObjectNode(id="far_obj", category="cup", pos=[0.5, 0.5], size=[0.2, 0.2], depth=0.8)
    node2 = ObjectNode(id="near_obj", category="cup", pos=[0.5, 0.5], size=[0.2, 0.2], depth=0.2)

    graph = PlanningGraph(objects=[node1, node2])
    stats = compute_instance_visible_areas(graph, resolution=(100, 100))

    # Node 2 completely occludes Node 1 because they share exact coordinates and Node 2 is nearer
    stat_dict = {item[0]: (item[1], item[2]) for item in stats}
    full_far, vis_far = stat_dict["far_obj"]
    full_near, vis_near = stat_dict["near_obj"]

    assert vis_near > 0
    assert vis_far == 0.0  # Completely covered
