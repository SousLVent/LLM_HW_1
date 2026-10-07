"""Check the normalization experiment against the archived baseline model."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest

EXERCISE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXERCISE_DIR.parent))

import torch
from model import GPT, GPTConfig, LayerNorm


def make_model(norm_type="layernorm"):
    config = GPTConfig(n_layer=6, n_head=6, n_embd=24, block_size=16,
                       vocab_size=65, dropout=0.0, bias=False, norm_type=norm_type)
    with contextlib.redirect_stdout(io.StringIO()):
        return GPT(config)


class NormalizationTests(unittest.TestCase):
    def test_layernorm_preserves_baseline_outputs(self):
        snapshot = EXERCISE_DIR / "results/shakespeare_char_baseline_20261004/source/model.py"
        spec = importlib.util.spec_from_file_location("baseline_model_snapshot", snapshot)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        torch.manual_seed(1337)
        current = make_model()
        args = vars(current.config).copy()
        args = {key: value for key, value in args.items()
                if key in baseline.GPTConfig.__dataclass_fields__}
        torch.manual_seed(1337)
        with contextlib.redirect_stdout(io.StringIO()):
            original = baseline.GPT(baseline.GPTConfig(**args))
        self.assertEqual(current.state_dict().keys(), original.state_dict().keys())
        for name, value in current.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)
        tokens = torch.arange(32).reshape(2, 16)
        current_logits, current_loss = current(tokens, tokens)
        original_logits, original_loss = original(tokens, tokens)
        torch.testing.assert_close(current_logits, original_logits, rtol=0, atol=0)
        torch.testing.assert_close(current_loss, original_loss, rtol=0, atol=0)

    def test_initial_weights_are_identical_between_norm_variants(self):
        torch.manual_seed(1337)
        layernorm = make_model("layernorm")
        torch.manual_seed(1337)
        rmsnorm = make_model("rmsnorm")
        self.assertEqual(layernorm.state_dict().keys(), rmsnorm.state_dict().keys())
        for name, value in layernorm.state_dict().items():
            torch.testing.assert_close(value, rmsnorm.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(layernorm.get_num_params(False), rmsnorm.get_num_params(False))

    def test_rmsnorm_replaces_all_thirteen_normalizations(self):
        model = make_model("rmsnorm")
        norms = [module for module in model.modules() if isinstance(module, torch.nn.RMSNorm)]
        self.assertEqual(len(norms), 13)
        self.assertFalse(any(isinstance(module, LayerNorm) for module in model.modules()))
        self.assertIsInstance(model.transformer.ln_f, torch.nn.RMSNorm)
        self.assertTrue(all(module.eps == 1e-5 for module in norms))

    def test_rmsnorm_keeps_the_mean_of_constant_input(self):
        layernorm = make_model("layernorm").transformer.ln_f
        rmsnorm = make_model("rmsnorm").transformer.ln_f
        values = torch.full((2, 3, 24), 2.0)
        torch.testing.assert_close(layernorm(values), torch.zeros_like(values))
        expected = torch.full_like(values, 2.0 / (4.0 + 1e-5) ** 0.5)
        torch.testing.assert_close(rmsnorm(values), expected)

    def test_checkpoint_round_trip_preserves_rmsnorm(self):
        model = make_model("rmsnorm")
        buffer = io.BytesIO()
        torch.save({"model_args": vars(model.config), "model": model.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        with contextlib.redirect_stdout(io.StringIO()):
            restored = GPT(GPTConfig(**checkpoint["model_args"]))
        restored.load_state_dict(checkpoint["model"])
        self.assertIsInstance(restored.transformer.ln_f, torch.nn.RMSNorm)
        tokens = torch.arange(32).reshape(2, 16)
        expected, _ = model(tokens, tokens)
        actual, loss = restored(tokens, tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in restored.parameters()))

    def test_legacy_model_args_default_to_layernorm(self):
        args = vars(make_model().config).copy()
        args.pop("norm_type")
        with contextlib.redirect_stdout(io.StringIO()):
            model = GPT(GPTConfig(**args))
        self.assertIsInstance(model.transformer.ln_f, LayerNorm)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
