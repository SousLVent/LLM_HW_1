"""Plot and compare the four completed architectural ablations with the baseline."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from compare_normalization import EXERCISE_DIR, read_run, plt
from run_architecture_experiments import ARCH_DEFAULTS, VARIANTS

LABELS = {"baseline": "Baseline", "swiglu": "SwiGLU", "nope": "NoPE", "rope": "RoPE", "gqa2": "GQA (group size 2)"}
COLORS = {"baseline": "#334155", "swiglu": "#d97706", "nope": "#059669", "rope": "#2563eb", "gqa2": "#9333ea"}
GROUPS = [("MLP: GELU vs. SwiGLU", ["baseline", "swiglu"], "mlp_comparison"),
          ("Positions: learned vs. NoPE vs. RoPE", ["baseline", "nope", "rope"], "position_comparison"),
          ("Attention: MHA vs. GQA", ["baseline", "gqa2"], "gqa_comparison")]


def draw_pair(axes, names, curves, title):
    for name in names:
        steps, train, val = zip(*curves[name])
        kwargs = dict(color=COLORS[name], linewidth=2, label=LABELS[name], marker="o", markersize=3)
        axes[0].plot(steps, train, **kwargs)
        axes[1].plot(steps, val, **kwargs)
        best = min(curves[name], key=lambda row: row[2])
        axes[1].scatter(best[0], best[2], color=COLORS[name], s=65,
                        edgecolor="white", linewidth=1, zorder=4)
    late = [r[2] for n in names for r in curves[n] if r[0] >= 500]
    axes[0].set(title=f"{title}\nTraining loss", xlim=(0, 5000), ylim=(0, 4.6))
    axes[1].set(title="Validation loss (from iteration 500)", xlim=(500, 5000),
                ylim=(min(late) - 0.045, max(late) + 0.045))
    for ax in axes:
        ax.set(xlabel="Training iteration", ylabel="Cross-entropy (nats / character)")
        ax.grid(color="#e2e8f0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)


def save_figure(fig, directory, stem):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="20261004")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.run_tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_tag):
        parser.error("run-tag must contain only letters, digits, underscores, or hyphens")
    output = args.output_dir or EXERCISE_DIR / "results" / f"architecture_comparison_{args.run_tag}"
    directories = {name: EXERCISE_DIR / "results" / f"shakespeare_char_{name}_{args.run_tag}"
                   for name in VARIANTS}
    directories = {"baseline": EXERCISE_DIR / "results/shakespeare_char_baseline_20261004", **directories}
    curves, summaries, metadata = {}, {}, {}
    try:
        for name, directory in directories.items():
            metadata[name] = json.loads((directory / "run_metadata.json").read_text())
            if metadata[name]["status"] != "completed":
                raise ValueError(f"{name}: training has not completed")
            curves[name], summaries[name] = read_run(directory)
        baseline = metadata["baseline"]
        baseline_config = {**ARCH_DEFAULTS, **baseline["config"]}
        # Count tied token/output weights once, for the original bias-free GPT.
        cfg = baseline["model_args"]
        if cfg["bias"]:
            raise ValueError("This comparison expects the original bias-free baseline")
        d = cfg["n_embd"]
        baseline_parameters = (cfg["vocab_size"] + cfg["block_size"]) * d + cfg["n_layer"] * (12 * d * d + 2 * d) + d
        summaries["baseline"]["total_parameters"] = baseline_parameters
        for name, overrides in VARIANTS.items():
            actual = metadata[name]
            expected_config = {**baseline_config, **overrides, "out_dir": actual["config"]["out_dir"]}
            if actual["config"] != expected_config or actual["data_sha256"] != baseline["data_sha256"]:
                raise ValueError(f"{name}: configuration or data differs beyond the intended ablation")
            summaries[name]["total_parameters"] = actual["total_parameters"]
        for name, summary in summaries.items():
            summary["best_checkpoint_validation_loss_full_precision"] = metadata[name]["best_checkpoint_validation_loss"]
            if summary["best_validation_iteration"] != metadata[name]["best_checkpoint_iteration"]:
                raise ValueError(f"{name}: log and checkpoint best iterations disagree")
            if abs(summary["best_validation_loss"] - metadata[name]["best_checkpoint_validation_loss"]) > 0.000051:
                raise ValueError(f"{name}: log and checkpoint best losses disagree")
            summary["best_validation_delta_vs_baseline"] = summary["best_validation_loss"] - summaries["baseline"]["best_validation_loss"]
            summary["best_validation_change_percent"] = 100 * summary["best_validation_delta_vs_baseline"] / summaries["baseline"]["best_validation_loss"]
            summary["final_validation_delta_vs_baseline"] = summary["final_validation_loss"] - summaries["baseline"]["final_validation_loss"]
            summary["parameter_change_percent"] = 100 * (summary["total_parameters"] / baseline_parameters - 1)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))

    output.mkdir(parents=True, exist_ok=True)
    comparison = {"runs": summaries, "loss_units": "nats per character", "seed": 1337,
                  "final_iteration": 5000, "evaluations_per_run": 21,
                  "design": "Independent ablations of the original LayerNorm/GELU/learned-position/MHA baseline.",
                  "limitation": "One seed per variant. Architectural changes also change random-number consumption during initialization; minibatch draws are not guaranteed identical.",
                  "timing_method": "Median logged iteration >=500, excluding evaluation iterations; approximate, not a controlled speed benchmark."}
    fixed_path = EXERCISE_DIR / "results" / f"architecture_comparison_{args.run_tag}" / "fixed_validation.json"
    if fixed_path.exists():
        fixed = json.loads(fixed_path.read_text())
        if fixed["validation_data_sha256"] != baseline["data_sha256"]["val.bin"]:
            parser.error("Shared-window validation uses different data")
        shared_metrics = {}
        shared_table = "| Variant | Validation loss on identical windows | Change vs. baseline |\n|---|---:|---:|\n"
        for name in directories:
            values = fixed["runs"][name]
            if values["checkpoint_iteration"] != summaries[name]["best_validation_iteration"]:
                parser.error(f"{name}: shared-window evaluation used a different checkpoint")
            shared_metrics[name] = {key: value for key, value in values.items() if key != "batch_losses"}
            shared_table += f"| {LABELS[name]} | {values['validation_loss']:.6f} | {values['change_percent_vs_baseline']:+.2f}% |\n"
        comparison["shared_batch_validation"] = {
            "window_seed": fixed["window_seed"], "eval_iters": fixed["eval_iters"],
            "batch_size": fixed["batch_size"], "block_size": fixed["block_size"],
            "compile": fixed["compile"], "runs": shared_metrics,
        }
        (output / "fixed_validation_metrics.md").write_text(shared_table)
    (output / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n")
    with (output / "loss_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration"] + [f"{name}_{split}_loss" for name in curves for split in ("train", "validation")])
        for index, row in enumerate(curves["baseline"]):
            writer.writerow([row[0]] + [value for name in curves for value in curves[name][index][1:]])

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(3, 2, figsize=(12, 12.5))
    for pair, (title, names, _) in zip(axes, GROUPS):
        draw_pair(pair, names, curves, title)
    fig.suptitle("Shakespeare architecture ablations", fontsize=17, fontweight="bold", y=0.996)
    caption = "Same default training settings and seed 1337; one architecture change per run.\nLosses: evaluation mode, 200 batches per split. Dots highlight best validation checkpoints."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.98), h_pad=2.8)
    save_figure(fig, output, "loss_comparison")
    for title, names, stem in GROUPS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        draw_pair(axes, names, curves, title)
        fig.text(0.5, 0.015, caption, ha="center", fontsize=9, color="#475569")
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        save_figure(fig, output, stem)

    headers = ["Variant", "Parameters", "Best validation", "Best iteration", "Final training", "Final validation", "Best change vs. baseline"]
    table = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    for name, values in summaries.items():
        table += (f"| {LABELS[name]} | {values['total_parameters']:,} | {values['best_validation_loss']:.4f} | "
                  f"{values['best_validation_iteration']:,} | {values['final_train_loss']:.4f} | "
                  f"{values['final_validation_loss']:.4f} | {values['best_validation_change_percent']:+.2f}% |\n")
    (output / "metrics.md").write_text(table)
    if not (output / "README.md").exists():
        methods = EXERCISE_DIR / "results/architecture_comparison_20261004/README.md"
        (output / "README.md").write_text(
            "# Architecture comparison\n\n" + table + "\n"
            "![Training and validation loss](loss_comparison.png)\n\n"
            f"[Experimental method]({os.path.relpath(methods, output)}) · "
            "[Raw data](loss_comparison.csv) · [Summary](comparison_summary.json)\n")

    for name in VARIANTS:
        directory = directories[name]
        subprocess.run([sys.executable, "-B", str(EXERCISE_DIR / "plot_training_loss.py"),
                        str(directory / "train.log"), "--title", f"Shakespeare character-level GPT: {LABELS[name]}"], check=True,
                       stdout=subprocess.DEVNULL)
        values = summaries[name]
        (directory / "README.md").write_text(
            f"# {LABELS[name]} Shakespeare experiment\n\n"
            f"Completed through iteration 5,000 with the default character-level Shakespeare settings and seed 1337. "
            f"Only `{metadata[name]['architecture_overrides']}` changes the baseline architecture; LayerNorm remains enabled.\n\n"
            f"Best validation: **{values['best_validation_loss']:.4f}**, iteration **{values['best_validation_iteration']}**. "
            f"Final training: **{values['final_train_loss']:.4f}**; final validation: **{values['final_validation_loss']:.4f}**. "
            f"Total trainable parameters: **{values['total_parameters']:,}**.\n\n"
            f"[Comparison report]({os.path.relpath(output / 'README.md', directory)}) · [Plot](loss_curve.png) · [PDF](loss_curve.pdf) · "
            "[Loss data](losses.csv) · [Log](train.log) · [Metadata](run_metadata.json) · [Best checkpoint](checkpoint/ckpt.pt)\n\n"
            "`source/` contains the model, trainer, and configurator used for this run. "
            "The checkpoint records the architecture so that `sample.py` loads the correct model automatically.\n")
    print(table)
    print(f"Saved comparison artifacts to {output}")


if __name__ == "__main__":
    main()
