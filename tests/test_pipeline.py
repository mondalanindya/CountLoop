"""Tests for the end-to-end CountLoopPipeline."""

import os
import shutil
import tempfile
import pytest

from countloop.config import CountLoopConfig
from countloop.pipeline import CountLoopPipeline


def test_pipeline_end_to_end_execution():
    temp_dir = tempfile.mkdtemp()
    try:
        config = CountLoopConfig(
            alpha=0.6,
            tau=0.85,
            max_rounds=3,
            seed=42,
            use_mock_engine=True,
            output_dir=temp_dir,
            verbose=False,
        )

        pipeline = CountLoopPipeline(config=config)
        result = pipeline.run(
            prompt="15 cups on a wooden table",
            target_count=15,
            output_dir=temp_dir,
        )

        assert result.target_count == 15
        assert result.iterations_run >= 1
        assert result.iterations_run <= 3
        assert result.composite_score > 0.0
        assert os.path.exists(result.final_image_path)
        assert os.path.exists(os.path.join(temp_dir, "planning_graph_final.json"))
        assert os.path.exists(os.path.join(temp_dir, "report.md"))
        assert len(result.history) == result.iterations_run

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
