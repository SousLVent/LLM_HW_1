"""Plot the paired evaluation losses printed by nanoGPT's train.py."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re

EXERCISE_DIR = Path(__file__).resolve().parent
DEFAULT_LOG = EXERCISE_DIR / "results/shakespeare_char_baseline_20261004/train.log"
os.environ.setdefault("MPLCONFIGDIR", str(EXERCISE_DIR / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, nargs="?", default=DEFAULT_LOG,
                        help="Training log; defaults to the saved Shakespeare baseline.")
    parser.add_argument("--output-dir", type=Path,
                        help="Output folder; defaults to the training log's folder.")
    parser.add_argument("--title", help="Optional plot title.")
    args = parser.parse_args()
    log = args.log.expanduser()
    if not log.is_absolute():
        for base in (Path.cwd(), EXERCISE_DIR, EXERCISE_DIR.parent):
            candidate = base / log
            if candidate.is_file():
                log = candidate
                break
    log = log.resolve()
    try:
        log_text = log.read_text(encoding="utf-8")
    except OSError as error:
        parser.error(f"Cannot read training log {log}: {error.strerror}")
    output = (args.output_dir.expanduser() if args.output_dir else log.parent).resolve()
    variants = ["RMSNorm"] if "normalization: rmsnorm;" in log_text else []
    architecture = re.search(
        r"architecture: mlp=(\w+); positions=(\w+); query_heads=(\d+); kv_heads=(\d+);", log_text)
    if architecture:
        mlp, positions, query_heads, kv_heads = architecture.groups()
        if mlp == "swiglu":
            variants.append("SwiGLU")
        if positions in {"nope", "rope"}:
            variants.append({"nope": "NoPE", "rope": "RoPE"}[positions])
        if query_heads != kv_heads:
            variants.append(f"GQA (group size {int(query_heads) // int(kv_heads)})")
    default_title = "Shakespeare character-level GPT: " + (", ".join(variants) or "baseline")

    pattern = re.compile(
        r"^step (\d+): train loss ([^,\s]+), val loss (\S+)$", re.MULTILINE
    )
    rows = [(int(step), float(train), float(val))
            for step, train, val in pattern.findall(log_text)]
    if not rows:
        parser.error("The log contains no paired training/validation evaluations.")
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        parser.error("Evaluation steps must be strictly increasing.")
    if not all(math.isfinite(value) for row in rows for value in row[1:]):
        parser.error("The log contains a non-finite loss.")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "losses.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "train_loss", "validation_loss"])
        writer.writerows(rows)

    steps, training, validation = zip(*rows)
    best = min(rows, key=lambda row: row[2])
    summary = {
        "evaluation_points": len(rows),
        "first_iteration": steps[0],
        "final_iteration": steps[-1],
        "final_train_loss": training[-1],
        "final_validation_loss": validation[-1],
        "best_validation_iteration": best[0],
        "best_validation_loss": best[2],
        "loss_units": "nats per character",
        "source": log.name,
        "precision": "Four decimal places, as printed by train.py",
    }
    (output / "loss_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(steps, training, color="#2563eb", linewidth=2,
            marker="o", markersize=3.5, label="Training")
    ax.plot(steps, validation, color="#d97706", linewidth=2,
            marker="o", markersize=3.5, label="Validation")
    ax.scatter([best[0]], [best[2]], color="#d97706", edgecolor="white",
               s=85, linewidth=1.5, zorder=4)
    ax.annotate(f"Best validation: {best[2]:.4f}\nIteration {best[0]:,}",
                xy=(best[0], best[2]), xytext=(-12, 35),
                textcoords="offset points", ha="right", fontsize=10,
                arrowprops={"arrowstyle": "-", "color": "#9ca3af"})
    ax.set(title=args.title or default_title,
           xlabel="Training iteration", ylabel="Cross-entropy loss (nats / character)")
    ax.set_xlim(steps[0], steps[-1] if len(rows) > 1 else steps[0] + 1)
    ax.set_ylim(bottom=0)
    ax.grid(axis="both", color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.5, 0.025,
             "Paired evaluation losses; dropout disabled on both splits.\n"
             "Default config: evaluations every 250 iterations, 200 batches per split.",
             ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"loss_curve.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Saved PNG: {output / 'loss_curve.png'}")
    print(f"Saved PDF: {output / 'loss_curve.pdf'}")
    print(f"Saved loss data: {output / 'losses.csv'}")


if __name__ == "__main__":
    main()
