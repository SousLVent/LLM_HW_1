"""Evaluate saved best checkpoints on exactly the same validation windows."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

EXERCISE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXERCISE_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.append(str(PROJECT_DIR.parent / "packages"))

import numpy as np
import torch
from model import GPT, GPTConfig
from run_architecture_experiments import VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="20261004")
    args = parser.parse_args()
    if not args.run_tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_tag):
        parser.error("run-tag must contain only letters, digits, underscores, or hyphens")
    if not torch.cuda.is_available():
        parser.error("CUDA access is required for this evaluation")
    os.environ.setdefault("CUDA_CACHE_PATH", str(EXERCISE_DIR / ".cache/cuda"))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    base = EXERCISE_DIR / "results"
    directories = {"baseline": base / "shakespeare_char_baseline_20261004",
                   **{name: base / f"shakespeare_char_{name}_{args.run_tag}" for name in VARIANTS}}
    val_path = PROJECT_DIR / "data/shakespeare_char/val.bin"
    val_hash = hashlib.sha256(val_path.read_bytes()).hexdigest()
    for name, directory in directories.items():
        metadata = json.loads((directory / "run_metadata.json").read_text())
        if metadata["status"] != "completed" or metadata["data_sha256"]["val.bin"] != val_hash:
            parser.error(f"{name}: run incomplete or validation data changed")

    batch_size, block_size, eval_iters, seed = 64, 256, 200, 20261004
    data = torch.from_numpy(np.fromfile(val_path, dtype=np.uint16).astype(np.int64))
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(len(data) - block_size, (eval_iters, batch_size), generator=generator)
    offsets = starts[..., None] + torch.arange(block_size)
    x, y = data[offsets].cuda(), data[offsets + 1].cuda()
    result = {
        "description": "Additional evaluation of best saved checkpoints on identical validation windows.",
        "window_seed": seed, "batch_size": batch_size, "block_size": block_size,
        "eval_iters": eval_iters, "evaluated_characters": eval_iters * batch_size * block_size,
        "window_starts_sha256": hashlib.sha256(starts.numpy().tobytes()).hexdigest(),
        "validation_data_sha256": val_hash, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16", "compile": False,
        "loss_units": "nats per character", "runs": {},
    }
    for name, directory in directories.items():
        checkpoint = torch.load(directory / "checkpoint/ckpt.pt", map_location="cpu", weights_only=True)
        net = GPT(GPTConfig(**checkpoint["model_args"]))
        net.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()})
        iteration = checkpoint["iter_num"]
        del checkpoint
        net = net.eval().cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            losses = torch.stack([net(x[i], y[i])[1] for i in range(eval_iters)])
        if not torch.isfinite(losses).all():
            raise RuntimeError(f"Non-finite validation loss: {name}")
        loss = losses.mean().item()
        result["runs"][name] = {"checkpoint_iteration": iteration, "validation_loss": loss,
                                "perplexity": float(np.exp(loss)),
                                "batch_losses": losses.cpu().tolist()}
        print(f"{name}: checkpoint {iteration}, fixed-window validation loss {loss:.6f}", flush=True)
        del net, losses
    baseline_loss = result["runs"]["baseline"]["validation_loss"]
    for metrics in result["runs"].values():
        metrics["delta_vs_baseline"] = metrics["validation_loss"] - baseline_loss
        metrics["change_percent_vs_baseline"] = 100 * metrics["delta_vs_baseline"] / baseline_loss
    output = base / f"architecture_comparison_{args.run_tag}" / "fixed_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
