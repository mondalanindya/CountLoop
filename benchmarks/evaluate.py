"""Benchmark evaluation metrics for CountLoop.

Computes:
- Mean Absolute Error (MAE): sum(|c_hat - c_gt|) / N
- Exact-Count Match Rate: % where c_hat == c_gt
- Tolerance Accuracy (+/- 5%, +/- 10%): % where |c_hat - c_gt| / c_gt <= tolerance
- Per-Tier Metrics (30-60, 61-120, 121-200)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def calculate_counting_metrics(
    eval_pairs: List[Tuple[int, int]],  # (c_hat, c_gt)
) -> Dict[str, Any]:
    """Calculates MAE, Exact Match, +/-5% and +/-10% tolerance accuracy."""
    if not eval_pairs:
        return {"mae": 0.0, "exact_rate": 0.0, "tol_5_rate": 0.0, "tol_10_rate": 0.0, "total": 0}

    total = len(eval_pairs)
    abs_errors = []
    exact_matches = 0
    tol_5_matches = 0
    tol_10_matches = 0

    # Stratified tiers: (30-60), (61-120), (121-200)
    tiers = {
        "30-60": {"errors": [], "exact": 0, "tol_5": 0, "tol_10": 0, "n": 0},
        "61-120": {"errors": [], "exact": 0, "tol_5": 0, "tol_10": 0, "n": 0},
        "121-200": {"errors": [], "exact": 0, "tol_5": 0, "tol_10": 0, "n": 0},
    }

    for c_hat, c_gt in eval_pairs:
        err = abs(c_hat - c_gt)
        rel_err = err / max(1.0, float(c_gt))
        abs_errors.append(err)

        if err == 0:
            exact_matches += 1
        if rel_err <= 0.05:
            tol_5_matches += 1
        if rel_err <= 0.10:
            tol_10_matches += 1

        # Tier classification
        tier_key = None
        if 30 <= c_gt <= 60:
            tier_key = "30-60"
        elif 61 <= c_gt <= 120:
            tier_key = "61-120"
        elif c_gt > 120:
            tier_key = "121-200"

        if tier_key:
            t = tiers[tier_key]
            t["n"] += 1
            t["errors"].append(err)
            if err == 0:
                t["exact"] += 1
            if rel_err <= 0.05:
                t["tol_5"] += 1
            if rel_err <= 0.10:
                t["tol_10"] += 1

    overall_mae = sum(abs_errors) / total
    exact_pct = (exact_matches / total) * 100.0
    tol_5_pct = (tol_5_matches / total) * 100.0
    tol_10_pct = (tol_10_matches / total) * 100.0

    tier_results = {}
    for k, v in tiers.items():
        if v["n"] > 0:
            tier_results[k] = {
                "mae": round(sum(v["errors"]) / v["n"], 2),
                "exact": f"{(v['exact'] / v['n']) * 100.0:.1f}%",
                "tol_5": f"{(v['tol_5'] / v['n']) * 100.0:.1f}%",
                "tol_10": f"{(v['tol_10'] / v['n']) * 100.0:.1f}%",
                "count": v["n"],
            }

    return {
        "mae": round(overall_mae, 2),
        "exact_rate": round(exact_pct, 1),
        "tol_5_rate": round(tol_5_pct, 1),
        "tol_10_rate": round(tol_10_pct, 1),
        "total_evaluated": total,
        "tiers": tier_results,
    }
