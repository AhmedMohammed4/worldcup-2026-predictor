"""
Brick 7: Calibration analysis.

From backtest predictions, bins probabilities and plots a reliability curve
(predicted vs actual), computes the Brier score, and breaks down calibration
by outcome type.

Usage:
    python calibration.py
"""

import math
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import run_backtest_data

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "worldcup.db"
PLOT_PATH = Path(__file__).resolve().parent.parent / "data" / "reliability.png"


def brier_score(predictions: list[dict]) -> float:
    """
    Multiclass Brier score for 1X2 predictions.
    Lower is better. Random (1/3, 1/3, 1/3) gives 0.667.
    """
    total = 0.0
    for p in predictions:
        outcome = p["outcome"]
        for side, prob in [("home", p["p_home"]), ("draw", p["p_draw"]), ("away", p["p_away"])]:
            actual = 1.0 if side == outcome else 0.0
            total += (prob - actual) ** 2
    return total / len(predictions)


def reliability_data(predictions: list[dict], n_bins: int = 10) -> dict:
    """
    Bin all individual outcome probabilities and compute predicted vs actual
    frequency per bin. Returns dict with bin_centers, pred_avg, actual_avg, counts.
    """
    # Collect all (predicted_prob, hit) pairs
    pairs = []
    for p in predictions:
        outcome = p["outcome"]
        for side, prob in [("home", p["p_home"]), ("draw", p["p_draw"]), ("away", p["p_away"])]:
            hit = 1.0 if side == outcome else 0.0
            pairs.append((prob, hit))

    pairs.sort(key=lambda x: x[0])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    pred_avg = []
    actual_avg = []
    counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [(pr, h) for pr, h in pairs if lo <= pr < hi or (i == n_bins - 1 and pr == hi)]
        if in_bin:
            probs, hits = zip(*in_bin)
            bin_centers.append(np.mean(probs))
            pred_avg.append(np.mean(probs))
            actual_avg.append(np.mean(hits))
            counts.append(len(in_bin))
        else:
            bin_centers.append((lo + hi) / 2)
            pred_avg.append((lo + hi) / 2)
            actual_avg.append(0)
            counts.append(0)

    return {
        "bin_centers": bin_centers,
        "pred_avg": pred_avg,
        "actual_avg": actual_avg,
        "counts": counts,
    }


def plot_reliability(rel: dict, brier: float, n_predictions: int):
    """Plot and save the reliability diagram."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[3, 1],
                                    gridspec_kw={"hspace": 0.3})

    centers = rel["pred_avg"]
    actuals = rel["actual_avg"]
    counts = rel["counts"]

    # Main reliability curve
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.plot(centers, actuals, "o-", color="#2563eb", markersize=8,
             linewidth=2, label="Model")
    ax1.set_xlabel("Predicted probability", fontsize=12)
    ax1.set_ylabel("Observed frequency", fontsize=12)
    ax1.set_title(f"Reliability Diagram (Brier={brier:.4f}, n={n_predictions})",
                  fontsize=13)
    ax1.legend(fontsize=11)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    # Histogram of prediction counts per bin
    bin_edges = np.linspace(0, 1, len(counts) + 1)
    bin_widths = np.diff(bin_edges)
    ax2.bar(bin_edges[:-1], counts, width=bin_widths, align="edge",
            color="#2563eb", alpha=0.6, edgecolor="white")
    ax2.set_xlabel("Predicted probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Prediction distribution", fontsize=11)
    ax2.set_xlim(-0.02, 1.02)

    plt.savefig(str(PLOT_PATH), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Reliability plot saved to {PLOT_PATH}")


def per_outcome_calibration(predictions: list[dict]):
    """Print calibration stats broken down by outcome type."""
    for side, key in [("Home win", "p_home"), ("Draw", "p_draw"), ("Away win", "p_away")]:
        probs = [p[key] for p in predictions]
        hits = [1.0 if p["outcome"] == key.split("_")[1] else 0.0 for p in predictions]

        # Map key to outcome string
        outcome_str = {"p_home": "home", "p_draw": "draw", "p_away": "away"}[key]
        hits = [1.0 if p["outcome"] == outcome_str else 0.0 for p in predictions]

        avg_pred = np.mean(probs)
        avg_actual = np.mean(hits)
        n = sum(hits)

        print(f"  {side:<10}: avg predicted={avg_pred:.3f}, actual rate={avg_actual:.3f}, "
              f"count={int(n)}/{len(predictions)}, gap={avg_pred - avg_actual:+.3f}")


def main():
    print("Running backtest to get predictions...\n")
    predictions = run_backtest_data()

    if not predictions:
        print("No predictions from backtest.")
        return

    print(f"\n{'=' * 60}")
    print("CALIBRATION REPORT")
    print(f"{'=' * 60}")

    # Brier score
    bs = brier_score(predictions)
    print(f"\nBrier score: {bs:.4f}")
    print(f"  (Random baseline: 0.6667, perfect: 0.0)")

    # Log loss (already computed, just average it)
    avg_ll = np.mean([p["log_loss"] for p in predictions])
    print(f"Log loss:    {avg_ll:.4f}")
    print(f"  (Random baseline: 1.0986)")

    # Per-outcome breakdown
    print(f"\nPer-outcome calibration:")
    per_outcome_calibration(predictions)

    # Reliability diagram
    print(f"\nReliability diagram:")
    rel = reliability_data(predictions, n_bins=10)
    plot_reliability(rel, bs, len(predictions))

    # Print bin details
    print(f"\n  {'Bin':>12} {'Predicted':>10} {'Actual':>10} {'Count':>8}")
    print(f"  {'-'*44}")
    for i in range(len(rel["bin_centers"])):
        if rel["counts"][i] > 0:
            print(f"  {rel['bin_centers'][i]:>11.0%} {rel['pred_avg'][i]:>10.3f} "
                  f"{rel['actual_avg'][i]:>10.3f} {rel['counts'][i]:>8}")

    print("\nDone.")


if __name__ == "__main__":
    main()
