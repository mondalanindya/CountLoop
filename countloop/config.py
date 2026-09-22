"""Configuration parameters and hyperparameters for CountLoop.

All default values precisely mirror those reported in the TMLR 2026 paper:
- alpha = 0.6, beta = 0.4
- tau = 0.85 (early stopping threshold)
- K = 3 (max refinement rounds)
- GroundingDINO confidence threshold = 0.3, NMS IoU = 0.5
- Area constraints: [1/100, 1/25]
- Max displacement: 0.08
- Seed: 42
- Resolution: 1024x1024
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CountLoopConfig:
    """Configuration class storing all model parameters and runtime settings."""

    # Scoring & Termination Hyperparameters
    alpha: float = 0.6  # Weight for count accuracy in composite score S
    beta: float = 0.4   # Weight for aesthetics (1.0 - alpha)
    tau: float = 0.85   # Quality threshold for early stopping
    max_rounds: int = 3 # Hard computational safeguard iteration cap K

    # Spatial & Layout Constraints (Design VLM & Psi)
    min_bbox_area: float = 0.01   # 1/100 of canvas area (~102x102 at 1024x1024)
    max_bbox_area: float = 0.04   # 1/25 of canvas area (~205x205 at 1024x1024)
    min_l2_dist: float = 0.03     # Minimum L2 distance between instance centers
    max_displacement: float = 0.08# Bounded per-round displacement for Psi refiner
    max_edits_per_round: int = 10 # Upper bound on typed edits per round

    # Detector & Evaluator Settings
    confidence_threshold: float = 0.3  # Detector score threshold for GroundingDINO
    nms_iou: float = 0.5               # Class-agnostic NMS IoU threshold
    min_visible_ratio: float = 0.1667  # 1/6: partially occluded visible area threshold

    # Diffusion & Image Generation
    resolution: Tuple[int, int] = (1024, 1024)
    num_inference_steps: int = 50       # Full schedule for final composition
    instance_inference_steps: int = 20  # Truncated schedule for per-instance pass
    guidance_scale: float = 7.5
    seed: int = 42

    # Model IDs & Paths
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    ip_adapter_checkpoint: Optional[str] = None
    detector_model_id: str = "IDEA-Research/grounding-dino-base"
    evaluator_model_id: str = "google/owlv2-base-patch16-ensemble"
    aesthetic_model_id: str = "q-align/q-align"

    # Runtime Engine Selection
    device: str = "auto"     # "cuda", "cpu", or "auto"
    dtype: str = "float16"   # "float16", "bfloat16", "float32"
    vlm_provider: str = "rule_based"  # "rule_based", "openai", "gemini", "hf", "mock"
    vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    api_key: Optional[str] = None
    use_mock_engine: bool = False  # Set to True for fast CPU/offline simulation

    # Output & Logging
    output_dir: str = "outputs"
    save_intermediate_steps: bool = True
    verbose: bool = True

    def __post_init__(self):
        # Validate weights
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        self.beta = round(1.0 - self.alpha, 4)

        # Resolve device
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
                self.use_mock_engine = True

        # Check API key from environment if not specified
        if self.api_key is None:
            self.api_key = (
                os.environ.get("OPENAI_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
            )
