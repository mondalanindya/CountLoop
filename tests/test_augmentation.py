"""Tests for data augmentation exporter."""

import os
import shutil
import tempfile
from PIL import Image

from countloop.augmentation import export_counting_annotations
from countloop.types import ObjectNode, PlanningGraph


def test_export_counting_annotations():
    temp_dir = tempfile.mkdtemp()
    try:
        nodes = [
            ObjectNode(id="cup_01", category="cup", pos=[0.3, 0.3], size=[0.1, 0.1], depth=0.2),
            ObjectNode(id="cup_02", category="cup", pos=[0.6, 0.6], size=[0.1, 0.1], depth=0.8),
        ]
        graph = PlanningGraph(objects=nodes)
        img = Image.new("RGB", (512, 512), color=(200, 200, 200))

        anno = export_counting_annotations(
            image=img,
            graph=graph,
            output_dir=temp_dir,
            image_name="test_aug.png",
            max_exemplars=2,
        )

        assert anno["image_file"] == "test_aug.png"
        assert anno["count"] == 2
        assert len(anno["points"]) == 2
        assert len(anno["boxes"]) == 2
        assert os.path.exists(os.path.join(temp_dir, "test_aug.png"))
        assert os.path.exists(os.path.join(temp_dir, "annotation.json"))
        assert len(anno["exemplar_crops"]) > 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
