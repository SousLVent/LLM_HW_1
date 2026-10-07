"""Behavioral checks for SwiGLU, NoPE, RoPE, and grouped-query attention."""

import contextlib
from dataclasses import asdict
import io
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from torch.nn import functional as F
from model import GPT, GPTConfig, MLP, CausalSelfAttention, apply_rotary

VARIANTS = {
    "baseline": {}, "swiglu": {"mlp_type": "swiglu"},
    "nope": {"pos_encoding": "nope"}, "rope": {"pos_encoding": "rope"},
    "gqa": {"kv_group_size": 2},
    "combined": {"norm_type": "rmsnorm", "mlp_type": "swiglu",
                 "pos_encoding": "rope", "kv_group_size": 2},
}


def config(**overrides):
    args = dict(n_layer=2, n_head=6, n_embd=24, block_size=16,
                vocab_size=65, dropout=0.0, bias=False)
    args.update(overrides)
    return GPTConfig(**args)


def model(**overrides):
    with contextlib.redirect_stdout(io.StringIO()):
        return GPT(config(**overrides))


class ArchitectureTests(unittest.TestCase):
    def test_swiglu_parameter_count_matches_default_mlp(self):
        original = MLP(config(n_embd=384))
        gated = MLP(config(n_embd=384, mlp_type="swiglu"))
        count = lambda module: sum(p.numel() for p in module.parameters())
        self.assertEqual(original.hidden_dim, 1536)
        self.assertEqual(gated.hidden_dim, 1024)
        self.assertEqual(count(original), 1179648)
        self.assertEqual(count(gated), count(original))

    def test_swiglu_uses_silu_gate_and_linear_value(self):
        mlp = MLP(config(mlp_type="swiglu")).double()
        x = torch.randn(2, 3, 24, dtype=torch.float64)
        gate_weight, value_weight = mlp.c_fc.weight.chunk(2)
        gate = F.linear(x, gate_weight)
        expected = F.linear(gate * torch.sigmoid(gate) * F.linear(x, value_weight),
                            mlp.c_proj.weight)
        torch.testing.assert_close(mlp(x), expected)

    def test_nope_and_rope_remove_position_parameters(self):
        original = model()
        for encoding in ("nope", "rope"):
            with self.subTest(encoding=encoding):
                variant = model(pos_encoding=encoding)
                self.assertNotIn("wpe", variant.transformer)
                self.assertEqual(original.get_num_params(False) - variant.get_num_params(False), 16 * 24)
                self.assertEqual(variant.get_num_params(), variant.get_num_params(False))

    def test_rotary_preserves_norms_and_relative_dot_products(self):
        attention = CausalSelfAttention(config(pos_encoding="rope"))
        q, k = torch.randn(2, 3, 5, 4), torch.randn(2, 3, 5, 4)
        rotate = lambda x, offset: apply_rotary(x, attention.rope_cos[offset:], attention.rope_sin[offset:])
        q0, k0, q5, k5 = rotate(q, 0), rotate(k, 0), rotate(q, 5), rotate(k, 5)
        torch.testing.assert_close(q0[..., 0, :], q[..., 0, :])
        torch.testing.assert_close(q0.norm(dim=-1), q.norm(dim=-1))
        torch.testing.assert_close(q0 @ k0.transpose(-1, -2), q5 @ k5.transpose(-1, -2), atol=2e-6, rtol=2e-6)

    def test_rope_matches_independent_complex_rotation_attention(self):
        attention = CausalSelfAttention(config(pos_encoding="rope")).double()
        x = torch.randn(2, 7, 24, dtype=torch.float64)
        q, k, v = [part.reshape(2, 7, 6, 4).transpose(1, 2)
                   for part in attention.c_attn(x).chunk(3, dim=-1)]
        angles = torch.outer(torch.arange(7, dtype=torch.float64), torch.tensor([1.0, 0.01]))
        phases = torch.polar(torch.ones_like(angles), angles)
        def rotate(tensor):
            pairs = torch.view_as_complex(tensor.reshape(2, 6, 7, 2, 2).contiguous())
            return torch.view_as_real(pairs * phases).flatten(-2)
        q, k = rotate(q), rotate(k)
        scores = (q @ k.transpose(-1, -2)) / 2
        scores = scores.masked_fill(torch.ones(7, 7, dtype=torch.bool).triu(1), -math.inf)
        expected = attention.c_proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 7, 24))
        torch.testing.assert_close(attention(x), expected, atol=2e-7, rtol=2e-6)

    def test_gqa_matches_mha_with_tied_key_value_heads_and_gradients(self):
        grouped = CausalSelfAttention(config(kv_group_size=2)).double()
        expanded = CausalSelfAttention(config()).double()
        q, k, v = grouped.c_attn.weight.split((24, 12, 12))
        repeat_heads = lambda weight: weight.reshape(3, 4, 24).repeat_interleave(2, dim=0).reshape(24, 24)
        with torch.no_grad():
            expanded.c_attn.weight.copy_(torch.cat((q, repeat_heads(k), repeat_heads(v))))
            expanded.c_proj.weight.copy_(grouped.c_proj.weight)
        self.assertEqual(grouped.n_kv_head, 3)
        x = torch.randn(2, 7, 24, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        actual, expected = grouped(x), expanded(y)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-10)
        grouped_grads = grouped.c_attn.weight.grad.split((24, 12, 12))
        expanded_grads = expanded.c_attn.weight.grad.chunk(3)
        for actual_grad, expanded_grad in zip(grouped_grads[1:], expanded_grads[1:]):
            summed = expanded_grad.reshape(3, 2, 4, 24).sum(1).reshape(12, 24)
            torch.testing.assert_close(actual_grad, summed, atol=1e-12, rtol=1e-10)

    def test_all_variants_are_causal_and_have_finite_gradients(self):
        tokens = torch.randint(65, (2, 12))
        modified = tokens.clone()
        modified[:, 6:] = (modified[:, 6:] + 17) % 65
        for name, overrides in VARIANTS.items():
            with self.subTest(variant=name):
                net = model(**overrides)
                original, loss = net(tokens, tokens)
                changed, _ = net(modified, modified)
                torch.testing.assert_close(original[:, :6], changed[:, :6], atol=0, rtol=0)
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_checkpoint_round_trip_crop_and_generation(self):
        tokens = torch.randint(65, (2, 12))
        for name, overrides in VARIANTS.items():
            with self.subTest(variant=name):
                net = model(**overrides).eval()
                buffer = io.BytesIO()
                torch.save({"model_args": asdict(net.config), "model": net.state_dict()}, buffer)
                buffer.seek(0)
                saved = torch.load(buffer, weights_only=True)
                restored = model(**saved["model_args"]).eval()
                restored.load_state_dict(saved["model"])
                torch.testing.assert_close(restored(tokens)[0], net(tokens)[0], rtol=0, atol=0)
                restored.crop_block_size(8)
                generated = restored.generate(tokens, max_new_tokens=3, top_k=5)
                self.assertEqual(generated.shape, (2, 15))

    def test_invalid_architecture_settings_fail_clearly(self):
        for overrides in ({"kv_group_size": 4}, {"kv_group_size": 0},
                          {"pos_encoding": "rope", "n_embd": 18},
                          {"pos_encoding": "rope", "rope_base": 0.0},
                          {"pos_encoding": "unknown"}, {"mlp_type": "unknown"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                model(**overrides)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
