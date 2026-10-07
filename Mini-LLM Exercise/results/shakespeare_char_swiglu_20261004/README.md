# SwiGLU Shakespeare experiment

Completed through iteration 5,000 with the default character-level Shakespeare settings and seed 1337. Only `{'mlp_type': 'swiglu'}` changes the baseline architecture; LayerNorm remains enabled.

Best validation: **1.4941**, iteration **1500**. Final training: **0.5190**; final validation: **1.8793**. Total trainable parameters: **10,745,088**.

[Comparison report](../architecture_comparison_20261004/README.md) · [Plot](loss_curve.png) · [PDF](loss_curve.pdf) · [Loss data](losses.csv) · [Log](train.log) · [Metadata](run_metadata.json) · [Best checkpoint](checkpoint/ckpt.pt)

`source/` contains the model, trainer, and configurator used for this run. The checkpoint records the architecture so that `sample.py` loads the correct model automatically.
