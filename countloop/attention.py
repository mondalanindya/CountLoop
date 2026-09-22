"""Layout-aligned attention masking and cumulative latent composition for CountLoop.

Implements the mathematical formulation of Section 3.2 and Supplementary Eq. 2-4:
- Spatial binary mask generation: M_i in {0, 1}^{W x H}
- Reshaping & interpolation: M_hat_i in {0, 1}^{h x w x 1}
- Self-segmentation contour refinement (k-means k=2 / IoM)
- Layout-aligned masked attention: A^i_mask = A^i_cross (x) M_hat_i
- Cumulative Latent Composition:
    F_{i+1}(x,y) = 1_{(x,y) in l_i} (x) A^i_mask + (1 - 1_{(x,y) in l_i}) (x) F_i
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np

from countloop.types import ObjectNode, PlanningGraph


def create_bbox_mask(
    bbox_xyxy: Tuple[float, float, float, float],
    height: int,
    width: int,
) -> np.ndarray:
    """Creates a 2D binary spatial mask M in {0, 1}^{height x width} for a normalized bbox."""
    x1_norm, y1_norm, x2_norm, y2_norm = bbox_xyxy

    x1 = int(round(max(0.0, min(1.0, x1_norm)) * width))
    y1 = int(round(max(0.0, min(1.0, y1_norm)) * height))
    x2 = int(round(max(0.0, min(1.0, x2_norm)) * width))
    y2 = int(round(max(0.0, min(1.0, y2_norm)) * height))

    mask = np.zeros((height, width), dtype=np.float32)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return mask


def self_segmentation_refinement(
    bbox_mask: np.ndarray,
    feature_map: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Refines rectangular binary mask into a shape-aware object contour.

    Follows Dahary et al. (2024):
    If feature map is provided, partitions the bounding box region using k-means
    (k=2) clustering to separate foreground object geometry from background.
    If no feature map is supplied, applies a smooth elliptical falloff contour.
    """
    if bbox_mask.sum() == 0:
        return bbox_mask.copy()

    refined = bbox_mask.copy()

    # If feature map is provided, perform k-means (k=2) foreground/background separation
    if feature_map is not None and feature_map.size > 0:
        # Extract features within mask
        indices = np.where(bbox_mask > 0.5)
        if len(indices[0]) > 4:
            region_features = feature_map[indices]
            # Simple 1D or multi-D k-means (k=2)
            c1 = region_features.min(axis=0)
            c2 = region_features.max(axis=0)
            for _ in range(5):
                d1 = np.linalg.norm(region_features - c1, axis=-1)
                d2 = np.linalg.norm(region_features - c2, axis=-1)
                cluster2 = d2 < d1
                if cluster2.sum() > 0:
                    c2 = region_features[cluster2].mean(axis=0)
                if (~cluster2).sum() > 0:
                    c1 = region_features[~cluster2].mean(axis=0)

            # Foreground cluster: cluster with higher norm/activation
            fg_mask = np.zeros_like(bbox_mask)
            fg_mask[indices] = (d2 < d1).astype(np.float32)
            if fg_mask.sum() > 0.1 * bbox_mask.sum():
                return fg_mask

    # Fallback shape contour: smooth rounded ellipse inscribed in the bounding box
    y_indices, x_indices = np.where(bbox_mask > 0.5)
    if len(y_indices) > 0:
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        cy = (y_min + y_max) / 2.0
        cx = (x_min + x_max) / 2.0
        ry = max(1.0, (y_max - y_min) / 2.0)
        rx = max(1.0, (x_max - x_min) / 2.0)

        Y, X = np.ogrid[: bbox_mask.shape[0], : bbox_mask.shape[1]]
        dist_sq = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        ellipse_mask = (dist_sq <= 1.0).astype(np.float32) * bbox_mask
        return ellipse_mask

    return refined


def apply_attention_masking(
    cross_attention_feature: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Applies layout-aligned attention masking:

    A^i_mask = A^i_cross (x) M_hat_i
    Confines cross-attention feature receptive field to the instance's spatial mask.
    """
    if mask.ndim == 2 and cross_attention_feature.ndim == 3:
        # Broadcast mask over channel dimension: (H, W, 1)
        expanded_mask = np.expand_dims(mask, axis=-1)
    else:
        expanded_mask = mask

    return cross_attention_feature * expanded_mask


def cumulative_latent_composition(
    f_prev: np.ndarray,
    a_mask: np.ndarray,
    indicator_mask: np.ndarray,
) -> np.ndarray:
    """Performs Cumulative Latent Composition according to Eq. 4 (main paper) and Eq. 8 (supp):

    F_{i+1}(x,y) = 1_{(x,y) in l_i} (x) A^i_mask + (1 - 1_{(x,y) in l_i}) (x) F_i

    Starting from F_0 = 0 in R^{H x W x D}. Instances are composed Far -> Near
    such that nearer objects naturally overwrite farther ones without feature distortion.
    """
    if indicator_mask.ndim == 2 and a_mask.ndim == 3:
        ind = np.expand_dims(indicator_mask, axis=-1)
    else:
        ind = indicator_mask

    return ind * a_mask + (1.0 - ind) * f_prev


def compute_instance_visible_areas(
    graph: PlanningGraph,
    resolution: Tuple[int, int] = (1024, 1024),
) -> List[Tuple[str, float, float]]:
    """Calculates full area and visible foreground area for every instance in Far -> Near order.

    Returns list of tuples: (instance_id, full_area_px, visible_area_px).
    """
    height, width = resolution
    # Compose Far -> Near (descending depth)
    ordered_nodes = graph.depth_sorted_instances()

    # Track pixel ownership canvas
    canvas = np.zeros((height, width), dtype=np.int32)
    full_areas: dict[str, float] = {}

    for idx, node in enumerate(ordered_nodes, start=1):
        x1, y1, x2, y2 = node.pixel_bbox(width, height)
        full_areas[node.id] = max(1.0, float((x2 - x1) * (y2 - y1)))
        # Overwrite canvas region with current instance ID
        canvas[y1:y2, x1:x2] = idx

    results = []
    for idx, node in enumerate(ordered_nodes, start=1):
        visible_px = float(np.sum(canvas == idx))
        full_px = full_areas[node.id]
        results.append((node.id, full_px, visible_px))

    return results


def check_occlusion_constraints(
    graph: PlanningGraph,
    min_visible_ratio: float = 0.1667,  # 1/6
    resolution: Tuple[int, int] = (1024, 1024),
) -> List[str]:
    """Identifies instances whose visible area is below the 1/6 visibility threshold."""
    violating_ids: List[str] = []
    stats = compute_instance_visible_areas(graph, resolution)

    for node_id, full_px, visible_px in stats:
        ratio = visible_px / full_px
        if ratio < min_visible_ratio:
            violating_ids.append(node_id)

    return violating_ids
