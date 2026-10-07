# Mini-LLM Exercise: LaTeX report

This standalone report packages all five exercise questions with their answers,
experimental methods, measured results, equations, and loss plots:

1. Baseline setup.
2. Effect of LayerNorm: pre-LayerNorm versus RMSNorm.
3. Effect of the MLP: GELU versus parameter-matched SwiGLU.
4. Effect of positional encoding: learned absolute positions, NoPE, and RoPE.
5. Effect of grouped-query attention with group size 2.

Files:

- `mini_llm_exercise.tex`: editable LaTeX source.
- `mini_llm_exercise.pdf`: compiled report.
- `figures/`: the five vector PDF figures; all image paths are relative.
- `data/`: saved numerical summaries and CSV data supporting the report.

The ZIP is self-contained and does not include model checkpoints or the training
environment. It can be uploaded to Overleaf; select `mini_llm_exercise.tex` as the
main document and pdfLaTeX as the compiler.

To compile locally, run from this folder:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build mini_llm_exercise.tex
```

The rebuilt report is `build/mini_llm_exercise.pdf`. Alternatively, run `pdflatex`
twice on `mini_llm_exercise.tex`. LaTeX packages are standard TeX Live packages;
no shell escape, network access, BibTeX, or external Python scripts are needed.

The report distinguishes best logged validation loss, final losses at iteration
5,000, and the additional evaluation on identical validation windows. RMSNorm was
not included in that additional shared-window evaluation. All reported training
results use one seed and should be interpreted accordingly.
