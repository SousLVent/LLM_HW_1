# Mini-LLM Exercise

Keep new exercise scripts, results, plots, logs, reports, and checkpoints in this folder.

## LaTeX report: all five questions

- [Compiled report](latex_report/mini_llm_exercise.pdf)
- [LaTeX source](latex_report/mini_llm_exercise.tex)
- [Portable ZIP with source, figures, data, and PDF](mini_llm_exercise_latex.zip)
- [Compilation instructions](latex_report/README.md)

## Baseline setup

The default Shakespeare character-level baseline is complete through iteration 5,000.

- [Baseline report](results/shakespeare_char_baseline_20261004/README.md)
- [Loss plot (PNG)](results/shakespeare_char_baseline_20261004/loss_curve.png)
- [Loss plot (PDF)](results/shakespeare_char_baseline_20261004/loss_curve.pdf)
- [Loss data (CSV)](results/shakespeare_char_baseline_20261004/losses.csv)
- [Best checkpoint](results/shakespeare_char_baseline_20261004/checkpoint/ckpt.pt)
- [Plotting script](plot_training_loss.py)

To regenerate the plot, run from `nanoGPT-master`:

```bash
python3 -B 'Mini-LLM Exercise/plot_training_loss.py'
```

You can also open `plot_training_loss.py` in VS Code and use **Run Python File**.
With no arguments, it finds the saved baseline log relative to the script, regardless
of the terminal's working directory. The PNG, PDF, and CSV are saved alongside that
log, and their paths are printed in the terminal.

To plot another training log, pass its path as the first argument. Relative log paths
are checked against the current directory, this exercise folder, and `nanoGPT-master`.

## Effect of normalization

The baseline uses **pre-LayerNorm**. The RMSNorm experiment replaces all 13
normalization layers while matching the baseline's training settings and seed.

- [Comparison report](results/layernorm_vs_rmsnorm_20261004/README.md)
- [Comparison plot](results/layernorm_vs_rmsnorm_20261004/loss_comparison.png)
- [Comparison data](results/layernorm_vs_rmsnorm_20261004/loss_comparison.csv)
- [RMSNorm run](results/shakespeare_char_rmsnorm_20261004/README.md)
- [Comparison script](compare_normalization.py) and [normalization checks](test_normalization.py)

To regenerate the comparison, run from `nanoGPT-master`:

```bash
python3 -B 'Mini-LLM Exercise/compare_normalization.py'
```

The comparison script also works with **Run Python File** in VS Code.

## Effects of MLP, positional encoding, and GQA

These are independent changes to the original LayerNorm baseline: parameter-matched
SwiGLU, no explicit positional encoding (NoPE), rotary positions (RoPE), and GQA
with two query heads per key/value head.

- [Implementation and comparison report](results/architecture_comparison_20261004/README.md)
- [Training and validation comparison](results/architecture_comparison_20261004/loss_comparison.png)
- [PDF plots](results/architecture_comparison_20261004/loss_comparison.pdf)
- [Comparison data](results/architecture_comparison_20261004/loss_comparison.csv)
- [Evaluation on identical validation windows](results/architecture_comparison_20261004/fixed_validation.json)
- [Experiment runner](run_architecture_experiments.py), [comparison script](compare_architectures.py),
  [checkpoint evaluation](evaluate_architecture_checkpoints.py), and [architecture checks](test_architecture_variants.py)

To regenerate the comparison after training:

```bash
python3 -B 'Mini-LLM Exercise/compare_architectures.py'
```

To run missing experiments, use `run_architecture_experiments.py`. It skips saved
completed runs; pass `--run-tag repeat1` to create a fresh set. New logs,
checkpoints, source snapshots, plots, and reports stay inside this folder.
