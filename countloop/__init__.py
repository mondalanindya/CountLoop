"""CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance (TMLR 2026)."""

from countloop.config import CountLoopConfig
from countloop.critic import CriticVLM
from countloop.design_vlm import DesignVLM
from countloop.generator import CountLoopGenerator
from countloop.pipeline import CountLoopPipeline
from countloop.refiner import TextualRefinementOperator
from countloop.types import (
    CriticFeedback,
    CriticScores,
    EditOperation,
    GenerationResult,
    IterationRecord,
    ObjectNode,
    PlanningGraph,
    SpatialRelation,
)

__version__ = "1.0.0"

__all__ = [
    "CountLoopPipeline",
    "CountLoopConfig",
    "PlanningGraph",
    "ObjectNode",
    "SpatialRelation",
    "CriticScores",
    "CriticFeedback",
    "EditOperation",
    "IterationRecord",
    "GenerationResult",
    "DesignVLM",
    "CriticVLM",
    "CountLoopGenerator",
    "TextualRefinementOperator",
]
