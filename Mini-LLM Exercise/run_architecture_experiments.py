"""Train independent Shakespeare ablations, saving every artifact in this folder."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

EXERCISE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXERCISE_DIR.parent
BASELINE_DIR = EXERCISE_DIR / "results/shakespeare_char_baseline_20261004"
VARIANTS = {
    "swiglu": {"mlp_type": "swiglu"},
    "nope": {"pos_encoding": "nope"},
    "rope": {"pos_encoding": "rope"},
    "gqa2": {"kv_group_size": 2},
}
ARCH_DEFAULTS = dict(norm_type="layernorm", mlp_type="gelu", pos_encoding="learned",
                     kv_group_size=1, rope_base=10000.0)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--run-tag", default="20261004",
                        help="Use a new tag for a fresh run; completed runs are skipped.")
    args = parser.parse_args()
    if not args.run_tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_tag):
        parser.error("run-tag must contain only letters, digits, underscores, or hyphens")
    pending = []
    for name in dict.fromkeys(args.variants):
        directory = EXERCISE_DIR / "results" / f"shakespeare_char_{name}_{args.run_tag}"
        if directory.exists():
            metadata_path = directory / "run_metadata.json"
            if metadata_path.exists() and json.loads(metadata_path.read_text()).get("status") == "completed":
                print(f"Already completed: {directory}", flush=True)
                continue
            parser.error(f"{directory} already exists and is incomplete; use a new --run-tag")
        pending.append((name, directory))
    if not pending:
        return

    import torch
    sys.path.insert(0, str(PROJECT_DIR))
    from model import GPT, GPTConfig
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable to this process. Run with access to the NVIDIA GPU.")
    baseline = json.loads((BASELINE_DIR / "run_metadata.json").read_text())
    data_dir = PROJECT_DIR / "data/shakespeare_char"
    data_hashes = {name: sha256(data_dir / name) for name in baseline["data_sha256"]}
    if data_hashes != baseline["data_sha256"]:
        parser.error("Dataset files differ from the recorded baseline")
    config_path = PROJECT_DIR / "config/train_shakespeare_char.py"
    if sha256(config_path) != baseline["source_sha256"]["config/train_shakespeare_char.py"]:
        parser.error("The default training config differs from the recorded baseline")
    gpu = torch.cuda.get_device_name(0)
    if gpu != baseline["gpu"] or torch.__version__ != baseline["torch_version"]:
        parser.error("GPU or PyTorch version differs from the recorded baseline")
    if not torch.cuda.is_bf16_supported():
        parser.error("The baseline requires CUDA bfloat16 support")

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for key, subdir in (("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                        ("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda"),
                        ("MPLCONFIGDIR", "matplotlib")):
        cache_dir = EXERCISE_DIR / ".cache" / subdir
        cache_dir.mkdir(parents=True, exist_ok=True)
        environment[key] = str(cache_dir)

    for name, directory in pending:
        directory.mkdir(parents=True)
        source_dir = directory / "source"
        source_dir.mkdir()
        sources = ["model.py", "train.py", "configurator.py", "config/train_shakespeare_char.py"]
        source_hashes = {source: sha256(PROJECT_DIR / source) for source in sources}
        for source in sources[:3]:
            shutil.copy2(PROJECT_DIR / source, source_dir / source)
        shutil.copy2(config_path, directory / "training_config.py")
        overrides = VARIANTS[name]
        output_dir = directory / "checkpoint"
        command = [sys.executable, "-u", "-B", str(PROJECT_DIR / "train.py"), str(config_path),
                   "--norm_type=layernorm", *[f"--{k}={v}" for k, v in overrides.items()],
                   f"--out_dir={output_dir}"]
        metadata = {
            "variant": name, "status": "running", "seed": 1337,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(command), "command_args": command,
            "artifact_directory": str(directory), "checkpoint_path": str(output_dir / "ckpt.pt"),
            "baseline_directory": str(BASELINE_DIR), "architecture_overrides": overrides,
            "config": {**baseline["config"], **ARCH_DEFAULTS, **overrides, "out_dir": str(output_dir)},
            "source_sha256": source_hashes, "data_sha256": data_hashes,
            "torch_version": torch.__version__, "gpu": gpu,
            "comparison_design": "One architectural change per run; original LayerNorm baseline; same training configuration and seed.",
        }
        metadata_path = directory / "run_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Starting {name}: {shlex.join(command)}", flush=True)
        started = time.monotonic()
        with (directory / "train.log").open("w", buffering=1) as log:
            process = subprocess.Popen(command, cwd=PROJECT_DIR, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                if line.startswith(("step ", "architecture:", "normalization:", "compiling")):
                    print(f"[{name}] {line.rstrip()}", flush=True)
            exit_code = process.wait()
        metadata.update(training_exit_code=exit_code, elapsed_seconds=time.monotonic() - started,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        if exit_code:
            metadata["status"] = "failed"
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print("\n".join((directory / "train.log").read_text().splitlines()[-30:]), flush=True)
            raise SystemExit(exit_code)
        checkpoint = torch.load(output_dir / "ckpt.pt", map_location="cpu", weights_only=True)
        net = GPT(GPTConfig(**checkpoint["model_args"]))
        net.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()})
        expected_config = metadata["config"]
        if checkpoint["config"] != expected_config:
            raise RuntimeError(f"Actual training configuration differs from expected: {name}")
        if not all(torch.isfinite(p).all() for p in net.parameters()):
            raise RuntimeError(f"Non-finite checkpoint parameters: {name}")
        metadata.update(status="completed", model_args=checkpoint["model_args"],
                        total_parameters=net.get_num_params(False),
                        mlp_hidden_dim=net.transformer.h[0].mlp.hidden_dim,
                        query_heads=net.config.n_head, kv_heads=net.transformer.h[0].attn.n_kv_head,
                        best_checkpoint_iteration=checkpoint["iter_num"],
                        best_checkpoint_validation_loss=float(checkpoint["best_val_loss"]),
                        checkpoint_verified=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Completed {name}: {metadata['elapsed_seconds']:.1f}s; "
              f"best validation={metadata['best_checkpoint_validation_loss']:.6f}; "
              f"parameters={metadata['total_parameters']:,}", flush=True)
        del checkpoint, net


if __name__ == "__main__":
    main()
