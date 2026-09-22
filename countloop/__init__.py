"""CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance (TMLR 2026).

Notice: The codebase is currently under preparation and will be fully released following publication.
"""

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
__status__ = "Under preparation; full public release following publication"

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
