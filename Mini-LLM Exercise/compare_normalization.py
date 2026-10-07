"""Compare the completed default LayerNorm and RMSNorm Shakespeare runs."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics

EXERCISE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(EXERCISE_DIR / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_run(directory):
    text = (directory / "train.log").read_text()
    pattern = re.compile(r"^step (\d+): train loss ([^,\s]+), val loss (\S+)$", re.MULTILINE)
    rows = [(int(step), float(train), float(val))
            for step, train, val in pattern.findall(text)]
    if [row[0] for row in rows] != list(range(0, 5001, 250)):
        raise ValueError(f"{directory.name}: expected 21 evaluations through iteration 5000")
    if not all(math.isfinite(value) for row in rows for value in row[1:]):
        raise ValueError(f"{directory.name}: non-finite loss")
    timings = [float(duration) for step, duration in re.findall(
        r"^iter (\d+): loss \S+, time ([\d.]+)ms,", text, re.MULTILINE
    ) if int(step) >= 500 and int(step) % 250 != 0]
    best = min(rows, key=lambda row: row[2])
    return rows, {
        "best_validation_loss": best[2],
        "best_validation_iteration": best[0],
        "train_loss_at_best_validation": best[1],
        "final_train_loss": rows[-1][1],
        "final_validation_loss": rows[-1][2],
        "best_validation_perplexity": math.exp(best[2]),
        "median_logged_iteration_ms": statistics.median(timings),
        "timing_samples": len(timings),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path,
                        default=EXERCISE_DIR / "results/shakespeare_char_baseline_20261004")
    parser.add_argument("--rms-dir", type=Path,
                        default=EXERCISE_DIR / "results/shakespeare_char_rmsnorm_20261004")
    parser.add_argument("--output-dir", type=Path,
                        default=EXERCISE_DIR / "results/layernorm_vs_rmsnorm_20261004")
    args = parser.parse_args()
    try:
        baseline, baseline_summary = read_run(args.baseline_dir)
        rmsnorm, rms_summary = read_run(args.rms_dir)
    except (OSError, ValueError, statistics.StatisticsError) as error:
        parser.error(str(error))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "loss_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "layernorm_train_loss", "layernorm_validation_loss",
                         "rmsnorm_train_loss", "rmsnorm_validation_loss"])
        for base, rms in zip(baseline, rmsnorm):
            writer.writerow([base[0], base[1], base[2], rms[1], rms[2]])

    delta = rms_summary["best_validation_loss"] - baseline_summary["best_validation_loss"]
    comparison = {
        "layernorm": baseline_summary,
        "rmsnorm": rms_summary,
        "rmsnorm_minus_layernorm_best_validation_loss": delta,
        "best_validation_loss_relative_change_percent": 100 * delta / baseline_summary["best_validation_loss"],
        "loss_units": "nats per character",
        "seeds": [1337],
        "timing_method": "Median logged iterations >=500, excluding evaluation iterations; approximate training-log timings.",
        "limitation": "One run per normalization type; this does not establish a consistent advantage across seeds.",
    }
    (output / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n")

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for label, rows, color in (("LayerNorm (baseline)", baseline, "#2563eb"),
                               ("RMSNorm", rmsnorm, "#d97706")):
        steps, train, val = zip(*rows)
        axes[0].plot(steps, train, label=label, color=color, linewidth=2,
                     marker="o", markersize=3)
        axes[1].plot(steps, val, label=label, color=color, linewidth=2,
                     marker="o", markersize=3)
        best = min(rows, key=lambda row: row[2])
        axes[1].scatter([best[0]], [best[2]], s=70, color=color,
                        edgecolor="white", linewidth=1.1, zorder=4)
    axes[0].set(title="Training loss", xlim=(0, 5000), ylim=(0, 4.5))
    late_values = [row[2] for rows in (baseline, rmsnorm) for row in rows if row[0] >= 500]
    axes[1].set(title="Validation loss (from iteration 500)", xlim=(500, 5000),
                ylim=(min(late_values) - 0.035, max(late_values) + 0.035))
    for ax in axes:
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Cross-entropy loss (nats / character)")
        ax.grid(color="#e5e7eb", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("LayerNorm vs. RMSNorm: default Shakespeare training", fontweight="bold", y=0.99)
    fig.text(0.5, 0.02,
             "Same configuration and seed (1337); all 13 normalization layers replaced; epsilon = 1e-5.\n"
             "Both losses use evaluation mode and 200 batches per split. Markers highlight the best validation loss.",
             ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"loss_comparison.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(comparison, indent=2))
    print(f"Saved comparison: {output / 'loss_comparison.png'}")


if __name__ == "__main__":
    main()
