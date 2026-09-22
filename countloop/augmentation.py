"""Data Augmentation module for Object Counting models (FSC-147 format).

Implements the synthetic data augmentation pipeline validated in Section 4.2
and Supplementary Section 5 of the TMLR 2026 paper:
- Generates point annotations (center coordinates)
- Generates bounding box annotations (pixel coordinates)
- Extracts 1-3 exemplar crops of visible foreground instances
- Produces FSC-147 and COCO compatible annotation dictionaries
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from countloop.attention import compute_instance_visible_areas
from countloop.types import ObjectNode, PlanningGraph


def export_counting_annotations(
    image: Image.Image,
    graph: PlanningGraph,
    output_dir: str,
    image_name: str = "image_001.png",
    max_exemplars: int = 3,
) -> Dict[str, Any]:
    """Exports self-labeled annotations and exemplar crops for object counting models."""
    os.makedirs(output_dir, exist_ok=True)
    width, height = image.size

    # Save base image
    img_path = os.path.join(output_dir, image_name)
    image.save(img_path)

    # Compute visible areas
    visible_stats = compute_instance_visible_areas(graph, (height, width))
    vis_dict = {stat[0]: stat[2] / stat[1] for stat in visible_stats}

    points: List[List[int]] = []
    boxes: List[List[int]] = []
    exemplar_boxes: List[List[int]] = []
    exemplar_paths: List[str] = []

    # Sort instances by visibility ratio descending to pick best exemplar candidates
    sorted_by_vis = sorted(
        graph.objects,
        key=lambda n: vis_dict.get(n.id, 0.0),
        reverse=True,
    )

    exemplars_saved = 0
    crops_dir = os.path.join(output_dir, "exemplar_crops")
    os.makedirs(crops_dir, exist_ok=True)

    for node in graph.objects:
        x1, y1, x2, y2 = node.pixel_bbox(width, height)
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))

        # Check if visible (>= 1/6 area)
        if vis_dict.get(node.id, 1.0) >= 0.1667:
            points.append([cx, cy])
            boxes.append([x1, y1, x2, y2])

    # Extract 1-3 exemplar crops from cleanest visible instances
    for node in sorted_by_vis:
        if exemplars_saved >= max_exemplars:
            break
        if vis_dict.get(node.id, 0.0) >= 0.70:  # High visibility exemplar
            x1, y1, x2, y2 = node.pixel_bbox(width, height)
            if x2 > x1 + 10 and y2 > y1 + 10:
                crop = image.crop((x1, y1, x2, y2))
                crop_name = f"exemplar_{exemplars_saved + 1}.png"
                crop_path = os.path.join(crops_dir, crop_name)
                crop.save(crop_path)
                exemplar_paths.append(crop_path)
                exemplar_boxes.append([x1, y1, x2, y2])
                exemplars_saved += 1

    annotation_data = {
        "image_file": image_name,
        "width": width,
        "height": height,
        "category": graph.objects[0].category if graph.objects else "object",
        "count": len(points),
        "points": points,
        "boxes": boxes,
        "box_examples_coordinates": exemplar_boxes,
        "exemplar_crops": exemplar_paths,
    }

    anno_path = os.path.join(output_dir, "annotation.json")
    with open(anno_path, "w", encoding="utf-8") as f:
        json.dump(annotation_data, f, indent=2)

    return annotation_data
