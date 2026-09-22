"""Command-line interface (CLI) for CountLoop."""

from __future__ import annotations

import argparse
import json
import os
import sys

from countloop.config import CountLoopConfig
from countloop.pipeline import CountLoopPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="countloop",
        description="CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: generate
    gen_parser = subparsers.add_parser("generate", help="Synthesize count-faithful images")
    gen_parser.add_argument("--prompt", "-p", type=str, required=True, help="Text prompt, e.g. '30 cups on a wooden table'")
    gen_parser.add_argument("--count", "-c", type=int, default=None, help="Explicit target count (optional, inferred from prompt if omitted)")
    gen_parser.add_argument("--output-dir", "-o", type=str, default="outputs", help="Output directory path")
    gen_parser.add_argument("--alpha", type=float, default=0.6, help="Weight for count accuracy (default: 0.6)")
    gen_parser.add_argument("--tau", type=float, default=0.85, help="Early stopping quality threshold (default: 0.85)")
    gen_parser.add_argument("--max-rounds", "-K", type=int, default=3, help="Max refinement rounds cap (default: 3)")
    gen_parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    gen_parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    gen_parser.add_argument("--mock", action="store_true", help="Force lightweight simulation mode")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI production server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    serve_parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate counting accuracy and aesthetics on an image")
    eval_parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    eval_parser.add_argument("--graph", "-g", type=str, required=True, help="Path to planning graph JSON")
    eval_parser.add_argument("--target-count", "-n", type=int, required=True, help="Target count N")

    # Command: augment
    aug_parser = subparsers.add_parser("augment", help="Generate self-labeled count training data (FSC-147 format)")
    aug_parser.add_argument("--prompt", "-p", type=str, required=True, help="Text prompt")
    aug_parser.add_argument("--count", "-c", type=int, default=None, help="Explicit target count")
    aug_parser.add_argument("--output-dir", "-o", type=str, default="augmented_data", help="Output directory path")
    aug_parser.add_argument("--mock", action="store_true", help="Force lightweight simulation mode")

    return parser


def handle_generate(args: argparse.Namespace) -> int:
    config = CountLoopConfig(
        alpha=args.alpha,
        tau=args.tau,
        max_rounds=args.max_rounds,
        seed=args.seed,
        device=args.device,
        use_mock_engine=args.mock,
        output_dir=args.output_dir,
    )

    pipeline = CountLoopPipeline(config=config)
    result = pipeline.run(
        prompt=args.prompt,
        target_count=args.count,
        output_dir=args.output_dir,
    )

    print(f"\nFinal Result:")
    print(f"  Image: {result.final_image_path}")
    print(f"  Detected Count: {result.final_count} / {result.target_count}")
    print(f"  Composite Score S: {result.composite_score:.4f}")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations_run}")
    print(f"  Total Time: {result.total_elapsed_sec:.2f} s")
    return 0


def handle_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
        from countloop.serve import app
        print(f"Starting CountLoop API server on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    except ImportError:
        print("Error: uvicorn is required for serving. Install with `pip install uvicorn`")
        return 1


def handle_evaluate(args: argparse.Namespace) -> int:
    from PIL import Image
    from countloop.critic import CriticVLM
    from countloop.types import PlanningGraph

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}")
        return 1
    if not os.path.exists(args.graph):
        print(f"Error: graph not found: {args.graph}")
        return 1

    img = Image.open(args.image)
    with open(args.graph, "r", encoding="utf-8") as f:
        graph = PlanningGraph.from_json(f.read())

    critic = CriticVLM()
    scores, feedback = critic.evaluate(img, graph, target_count=args.target_count)

    print("\nEvaluation Results:")
    print(f"  Detected: {scores.detected_count} / {scores.target_count}")
    print(f"  Normalized Count Score s_c: {scores.s_c:.4f}")
    print(f"  Aesthetic Score s_a: {scores.s_a:.4f}")
    print(f"  Composite Score S: {scores.S:.4f}")
    print(f"  Summary: {feedback.feedback.get('summary')}")
    print(f"  Edits suggested: {len(feedback.edits)}")
    return 0


def handle_augment(args: argparse.Namespace) -> int:
    from PIL import Image
    from countloop.augmentation import export_counting_annotations

    config = CountLoopConfig(
        use_mock_engine=args.mock,
        output_dir=args.output_dir,
    )
    pipeline = CountLoopPipeline(config=config)
    result = pipeline.run(
        prompt=args.prompt,
        target_count=args.count,
        output_dir=args.output_dir,
    )

    img = Image.open(result.final_image_path)
    anno = export_counting_annotations(
        image=img,
        graph=result.planning_graph,
        output_dir=args.output_dir,
    )

    print("\nData Augmentation Export Complete:")
    print(f"  Image: {anno['image_file']}")
    print(f"  Labeled Instances: {anno['count']}")
    print(f"  Point Annotations: {len(anno['points'])}")
    print(f"  Bounding Boxes: {len(anno['boxes'])}")
    print(f"  Exemplar Crops: {len(anno['exemplar_crops'])}")
    print(f"  Saved to: {os.path.abspath(args.output_dir)}")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        sys.exit(handle_generate(args))
    elif args.command == "serve":
        sys.exit(handle_serve(args))
    elif args.command == "evaluate":
        sys.exit(handle_evaluate(args))
    elif args.command == "augment":
        sys.exit(handle_augment(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
