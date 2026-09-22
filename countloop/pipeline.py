"""CountLoop Pipeline coordinating the full Agentic Synthesis-Critique-Refine Loop.

Direct implementation of Algorithm 1 from:
"CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance" (TMLR 2026).
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from PIL import Image

from countloop.config import CountLoopConfig
from countloop.critic import CriticVLM
from countloop.design_vlm import DesignVLM
from countloop.generator import CountLoopGenerator
from countloop.refiner import TextualRefinementOperator
from countloop.types import (
    CriticFeedback,
    CriticScores,
    GenerationResult,
    IterationRecord,
    PlanningGraph,
)


class CountLoopPipeline:
    """Master pipeline managing the iterative agentic generation loop."""

    def __init__(self, config: Optional[CountLoopConfig] = None):
        self.config = config or CountLoopConfig()
        self.design_vlm = DesignVLM(self.config)
        self.generator = CountLoopGenerator(self.config)
        self.critic = CriticVLM(self.config)
        self.refiner = TextualRefinementOperator(self.config)

    def run(
        self,
        prompt: str,
        target_count: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> GenerationResult:
        """Executes the complete CountLoop workflow according to Algorithm 1.

        Args:
            prompt: Text prompt with explicit or implicit numeric target.
            target_count: Optional explicit integer target count.
            output_dir: Directory where generated images and logs are stored.

        Returns:
            GenerationResult containing final image, final graph, metrics, and trajectory.
        """
        start_time = time.time()
        out_dir = output_dir or self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # Infer target count if not provided
        if target_count is None:
            parsed = self.design_vlm.parse_prompt_entities(prompt)
            target_count = sum(cnt for _, cnt in parsed) if parsed else 1

        if self.config.verbose:
            print(f"\n{'='*70}\n[CountLoop] Starting generation for: '{prompt}' (Target N={target_count})\n{'='*70}")

        # ----------------------------------------------------------------------
        # Stage 1: Plan (Design VLM -> Planning Graph G_0)
        # ----------------------------------------------------------------------
        plan_t0 = time.time()
        g_current = self.design_vlm.plan_layout(prompt, target_count=target_count)
        if self.config.verbose:
            print(f"[Stage 1: Plan] Planning graph built with {g_current.num_instances} instances ({time.time() - plan_t0:.2f}s)")

        # Save initial planning graph
        with open(os.path.join(out_dir, "planning_graph_init.json"), "w", encoding="utf-8") as f:
            f.write(g_current.to_json())

        history: List[IterationRecord] = []
        edited_ids: Optional[set[str]] = None
        converged = False
        k = 0
        final_image_path = ""

        # ----------------------------------------------------------------------
        # Stage 2: Iterative Synthesize-Critique-Refine Loop
        # ----------------------------------------------------------------------
        while k < self.config.max_rounds:
            round_t0 = time.time()
            if self.config.verbose:
                print(f"\n--- [Round {k}] ---")

            # 1. Synthesize
            round_img_path = os.path.join(out_dir, f"round_{k}.png")
            img = self.generator.generate(
                graph=g_current,
                edited_target_ids=edited_ids,
                output_path=round_img_path,
            )
            final_image_path = round_img_path

            # 2. Critique
            scores, feedback = self.critic.evaluate(img, g_current, target_count=target_count)

            if self.config.verbose:
                print(
                    f"[Critic] Detected: {scores.detected_count}/{scores.target_count} | "
                    f"s_c: {scores.s_c:.3f} | s_a: {scores.s_a:.3f} | S: {scores.S:.3f} "
                    f"(tau={self.config.tau:.2f})"
                )

            # Record round
            record = IterationRecord(
                round_idx=k,
                scores=scores,
                planning_graph=g_current.model_copy(deep=True),
                critic_feedback=feedback,
                image_path=round_img_path,
                elapsed_time_sec=round(time.time() - round_t0, 2),
            )

            # 3. Check early stopping condition: S >= tau
            if scores.S >= self.config.tau:
                converged = True
                if self.config.verbose:
                    print(f"[CountLoop] Early stopping reached! S={scores.S:.3f} >= tau={self.config.tau:.2f}")
                history.append(record)
                break

            # If reached hard cap K, stop
            if k + 1 >= self.config.max_rounds:
                if self.config.verbose:
                    print(f"[CountLoop] Hard cap K={self.config.max_rounds} reached.")
                history.append(record)
                break

            # 4. Refine (Parameter-Free Textual Operator Psi)
            g_next, trace, edited_ids = self.refiner.refine(g_current, feedback)
            record.reasoning_trace = trace
            history.append(record)

            if self.config.verbose and trace:
                print(f"[Refiner Psi]\n  {trace}")

            g_current = g_next
            k += 1

        total_elapsed = round(time.time() - start_time, 2)
        last_scores = history[-1].scores if history else None
        final_count = last_scores.detected_count if last_scores else 0
        composite_score = last_scores.S if last_scores else 0.0

        # Save final artifacts
        final_image_dest = os.path.join(out_dir, "final_image.png")
        if os.path.exists(final_image_path):
            img = Image.open(final_image_path)
            img.save(final_image_dest)

        with open(os.path.join(out_dir, "planning_graph_final.json"), "w", encoding="utf-8") as f:
            f.write(g_current.to_json())

        # Write summary report markdown
        self._write_report(out_dir, prompt, target_count, final_count, composite_score, converged, len(history), total_elapsed, history)

        result = GenerationResult(
            prompt=prompt,
            target_count=target_count,
            final_count=final_count,
            composite_score=composite_score,
            converged=converged,
            iterations_run=len(history),
            final_image_path=final_image_dest,
            planning_graph=g_current,
            history=history,
            total_elapsed_sec=total_elapsed,
        )

        if self.config.verbose:
            print(f"\n{'='*70}\n[CountLoop Finished] Prompt: '{prompt}' | Count: {final_count}/{target_count} | S: {composite_score:.3f} | Elapsed: {total_elapsed}s\n{'='*70}\n")

        return result

    def _write_report(
        self,
        out_dir: str,
        prompt: str,
        target_count: int,
        final_count: int,
        composite_score: float,
        converged: bool,
        rounds: int,
        elapsed: float,
        history: List[IterationRecord],
    ) -> None:
        """Writes trajectory markdown report."""
        report_path = os.path.join(out_dir, "report.md")
        lines = [
            "# CountLoop Execution Report",
            "",
            f"- **Prompt**: `{prompt}`",
            f"- **Target Count ($c_{{gt}}$)**: {target_count}",
            f"- **Final Detected Count ($\\hat{{c}}$)**: {final_count}",
            f"- **Final Composite Score ($S$)**: {composite_score:.4f} (threshold $\\tau = {self.config.tau}$)",
            f"- **Converged**: {'Yes (Early Stopping)' if converged else 'No (Reached Max Rounds)'}",
            f"- **Total Iterations**: {rounds}",
            f"- **Total Wall-Clock Time**: {elapsed:.2f} s",
            "",
            "## Iteration Trajectory",
            "",
            "| Round | Detected Count | $s_c$ | $s_a$ | Composite $S$ | Edits | Elapsed |",
            "|---|---|---|---|---|---|---|",
        ]

        for rec in history:
            s = rec.scores
            num_edits = len(rec.critic_feedback.edits)
            lines.append(
                f"| {rec.round_idx} | {s.detected_count} / {s.target_count} | {s.s_c:.3f} | {s.s_a:.3f} | {s.S:.3f} | {num_edits} | {rec.elapsed_time_sec:.2f}s |"
            )

        lines.extend([
            "",
            "## Reasoning Traces",
            "",
        ])
        for rec in history:
            if rec.reasoning_trace:
                lines.append(f"### Round {rec.round_idx} Refinement")
                lines.append(f"```\n{rec.reasoning_trace}\n```")
                lines.append("")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
