"""Aggregate the LLM-as-judge scores into a per-dimension comparison.

Reads outputs/eval_scores.json, prints a table of mean scores (per model x
dimension + overall), and saves a grouped bar chart to assets/eval_scores.png.
The per-dimension view is deliberate: it shows each model's PROFILE (strengths
and weaknesses) rather than a single winner.

Run from the repo root:  python scripts/eval_report.py
"""
import json
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DIMENSIONS = ["grammar", "fluency", "coherence",
              "contextual_correctness", "creativity", "plot_completion"]


def main():
    scored = json.load(open(REPO / "outputs" / "eval_scores.json"))

    # group scores by model -> dimension -> list of values
    models = []
    per_model = {}
    for r in scored:
        m = r["model"]
        if m not in per_model:
            per_model[m] = {d: [] for d in DIMENSIONS}
            models.append(m)
        for d in DIMENSIONS:
            per_model[m][d].append(float(r["scores"][d]))

    # ---- print table ----
    header = f"{'model':<18}" + "".join(f"{d[:10]:>12}" for d in DIMENSIONS) + f"{'OVERALL':>10}"
    print(header)
    print("-" * len(header))
    means = {}
    for m in models:
        row_means = [mean(per_model[m][d]) for d in DIMENSIONS]
        means[m] = row_means
        overall = mean(row_means)
        print(f"{m:<18}" + "".join(f"{v:>12.2f}" for v in row_means) + f"{overall:>10.2f}")

    # ---- grouped bar chart ----
    x = np.arange(len(DIMENSIONS))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(models):
        ax.bar(x + i * width, means[m], width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([d.replace("_", "\n") for d in DIMENSIONS])
    ax.set_ylabel("mean judge score (1-5)")
    ax.set_ylim(0, 5)
    ax.set_title("TinyStories eval — Llama-3.3-70B judge, per dimension")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(REPO / "assets" / "eval_scores.png", dpi=150, bbox_inches="tight")
    print("\nsaved chart to assets/eval_scores.png")


if __name__ == "__main__":
    main()
