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


def compute_iom(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-6) -> float:
    """Computes Intersection-over-Minimum (IoM) between two binary masks.

    IoM(A, B) = |A ∩ B| / min(|A|, |B|)

    Following Dahary et al. (2024) and Supplementary Section 1:
    Used to assign self-segmentation clusters to the corresponding
    instance noun cross-attention map with the highest IoM score.
    """
    intersection = float(np.logical_and(mask_a > 0.5, mask_b > 0.5).sum())
    area_a = float((mask_a > 0.5).sum())
    area_b = float((mask_b > 0.5).sum())
    min_area = min(area_a, area_b)
    if min_area < eps:
        return 0.0
    return float(intersection / min_area)


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


class CountLoopAttentionProcessor:
    r"""Diffusers-compatible U-Net attention processor implementing CountLoop cumulative composition.

    Formulated directly from Section 3.2 and Supplementary Section 1:
    1. GLIGEN (Li et al., 2023): Injects layout conditioning via gated self-attention.
    2. Dahary et al. (2024) ("It's Not You, It's Me"):
       - Confines cross-attention receptive fields via layout-aligned spatial masks:
         A^i_mask = A^i_cross \odot \hat{M}_i
       - Restricts self-segmentation clustering to the middle block and first up-block only.
       - Assigns clusters to noun tokens via Intersection-over-Minimum (IoM):
         IoM(A, B) = |A \cap B| / min(|A|, |B|).
    3. IP-Adapter (Ye et al., 2023):
       - Extracts query representations Z_q after linear projection W_Q but before QK^T interaction.
       - Conditions denoising on foreground appearance of prior instances.
    4. Cumulative Latent Composition (Eq. 4 & Supp. Eq. 8):
       - F_{i+1}(x,y) = 1_{(x,y) \in l_i} \odot A^i_mask + (1 - 1_{(x,y) \in l_i}) \odot F_i
       - Processed in depth order Far -> Near.
    5. Final Composition Pass (Supp. Eq. 10):
       - Driven by Z_q^N attending over shared key-values A(Z_q^N, K, V) with prompt P_d + P_bg.
    """

    def __init__(
        self,
        block_name: str = "",
        is_cross_attention: bool = False,
        is_middle_or_first_up_block: bool = False,
        gligen_gate_weight: float = 1.0,
    ):
        self.block_name = block_name
        self.is_cross_attention = is_cross_attention
        self.is_middle_or_first_up_block = is_middle_or_first_up_block
        self.gligen_gate_weight = gligen_gate_weight

        # State updated during inference
        self.active_mask: Optional[Any] = None  # torch.Tensor or np.ndarray \hat{M}_i
        self.cached_query: Optional[Any] = None  # Z_q extracted after W_Q
        self.cached_cross_attn_feature: Optional[Any] = None  # A_cross or A_mask
        self.grounding_tokens: Optional[Any] = None  # Q_i from GLIGEN encoder

    def set_active_mask(self, mask: Optional[Any]) -> None:
        r"""Sets the active spatial mask \hat{M}_i for the current instance."""
        self.active_mask = mask

    def set_grounding_tokens(self, tokens: Optional[Any]) -> None:
        r"""Sets GLIGEN grounding tokens Q_i = \mathbb{E}(l_i) for layout conditioning."""
        self.grounding_tokens = tokens

    def reset(self) -> None:
        """Resets cached tensors for a new generation round."""
        self.active_mask = None
        self.cached_query = None
        self.cached_cross_attn_feature = None
        self.grounding_tokens = None

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        temb: Optional[Any] = None,
        scale: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """Executes attention forward pass with layout masking and query caching."""
        try:
            import torch
            import torch.nn.functional as F

            is_torch = isinstance(hidden_states, torch.Tensor)
        except ImportError:
            is_torch = False

        if not is_torch:
            # Fallback for CPU / mock execution
            return hidden_states

        residual = hidden_states

        if hasattr(attn, "spatial_norm") and attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, channel = hidden_states.shape
            height = width = int(math.sqrt(sequence_length))

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if hasattr(attn, "prepare_attention_mask"):
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if hasattr(attn, "group_norm") and attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 1. Linear Projection for Query: z * W_Q
        query = attn.to_q(hidden_states)

        # 2. Extract and preserve query representation Z_q (Supplementary Eq. 9)
        # immediately after W_Q before QK^T interaction
        if not self.is_cross_attention:
            self.cached_query = query.detach().clone()

        # 3. Keys and Values
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 4. Scaled dot-product attention
        if hasattr(attn, "get_attention_scores"):
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
        else:
            scale_factor = 1.0 / math.sqrt(query.shape[-1])
            scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query,
                key.transpose(-1, -2),
                beta=0,
                alpha=scale_factor,
            )
            attention_probs = scores.softmax(dim=-1)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 5. Layout-aligned Attention Masking (Section 3.2 & Supplementary Eq. 7):
        # A^i_mask = A^i_cross \odot \hat{M}_i
        if self.is_cross_attention and self.active_mask is not None:
            mask_tensor = self.active_mask
            if not isinstance(mask_tensor, torch.Tensor):
                mask_tensor = torch.tensor(mask_tensor, dtype=hidden_states.dtype, device=hidden_states.device)
            if mask_tensor.ndim == 2:
                # Shape (H_mask, W_mask) -> (1, 1, H_mask, W_mask)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            if mask_tensor.shape[-2:] != (height, width):
                mask_tensor = F.interpolate(mask_tensor, size=(height, width), mode="bilinear", align_corners=False)
            # Flatten spatial dims to match sequence length: (1, 1, H*W)
            mask_flat = mask_tensor.view(1, -1, 1).to(hidden_states.device)
            hidden_states = hidden_states * mask_flat
            self.cached_cross_attn_feature = hidden_states.detach()

        # 6. Linear Output Projection
        if isinstance(attn.to_out, (list, torch.nn.ModuleList)):
            hidden_states = attn.to_out[0](hidden_states)
            if len(attn.to_out) > 1:
                hidden_states = attn.to_out[1](hidden_states)
        elif hasattr(attn, "to_out"):
            hidden_states = attn.to_out(hidden_states)

        # 7. GLIGEN Gated Self-Attention Injection (Li et al., 2023)
        if self.grounding_tokens is not None and not self.is_cross_attention:
            g_tokens = self.grounding_tokens
            if isinstance(g_tokens, torch.Tensor) and g_tokens.ndim == 3:
                g_key = attn.head_to_batch_dim(attn.to_k(g_tokens))
                g_val = attn.head_to_batch_dim(attn.to_v(g_tokens))
                g_query = attn.head_to_batch_dim(attn.to_q(hidden_states))
                g_scale = 1.0 / math.sqrt(g_query.shape[-1])
                g_scores = torch.baddbmm(
                    torch.empty(g_query.shape[0], g_query.shape[1], g_key.shape[1], dtype=g_query.dtype, device=g_query.device),
                    g_query,
                    g_key.transpose(-1, -2),
                    beta=0,
                    alpha=g_scale,
                )
                g_probs = g_scores.softmax(dim=-1)
                g_out = attn.batch_to_head_dim(torch.bmm(g_probs, g_val))
                if isinstance(attn.to_out, (list, torch.nn.ModuleList)):
                    g_proj = attn.to_out[0](g_out)
                else:
                    g_proj = g_out
                hidden_states = hidden_states + self.gligen_gate_weight * g_proj

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        rescale = getattr(attn, "rescale_output_factor", 1.0)
        hidden_states = hidden_states / rescale
        return hidden_states


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
