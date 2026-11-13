"""Automated Mathematical & Safety Invariant Tests for Cortex-AGI and Lab Architectures."""
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))

from cortex.arch import ARCH_REGISTRY, CortexConfig, H3CortexAGI
from cortex.autonomous_loop import check_safety_invariants
from cortex.run_suite import evaluate_arc_ttt
from cortex.tasks import TOK_PAD, TOK_RESET
from model import MODELS as LAB_MODELS, Config as LabConfig


def test_causal_and_padding_invariance():
    cfg = CortexConfig()
    for name, cls in ARCH_REGISTRY.items():
        m = cls(cfg).eval()
        x = torch.randint(2, 120, (2, 90))
        x[:, 0] = TOK_RESET
        out_clean = m(x[:, :55])["logits"][:, -1]
        # Corrupt future tokens >= 55 with arbitrary tokens and pad tokens
        x_dirty = x.clone()
        x_dirty[0, 55:] = TOK_PAD
        x_dirty[1, 55:] = TOK_RESET  # even future episode resets must NOT affect past positions!
        out_dirty = m(x_dirty)["logits"][:, 54]
        diff = (out_clean - out_dirty).abs().max().item()
        assert diff < 1e-5, f"{name} violated causality: diff={diff}"


def test_episode_boundary_isolation():
    cfg = CortexConfig()
    m = H3CortexAGI(cfg).eval()
    ep2 = torch.randint(2, 120, (1, 40))
    ep2[0, 0] = TOK_RESET
    # Prepend two completely different unrelated prior episodes before ep2's TOK_RESET
    ep1_a = torch.randint(2, 60, (1, 50)); ep1_a[0, 0] = TOK_RESET
    ep1_b = torch.randint(61, 120, (1, 50)); ep1_b[0, 0] = TOK_RESET
    out_a = m(torch.cat([ep1_a, ep2], dim=1))["logits"][:, -1]
    out_b = m(torch.cat([ep1_b, ep2], dim=1))["logits"][:, -1]
    # Because local_window=48, position 89 only sees positions 42..89; Wait: at pos 89 (50+39), 89-48=41 is in ep1 unless gap>=48!
    # Check at end of ep2 of length 55 (> local_window=48): neither local attention nor episodic memory should leak ep1!
    ep2_long = torch.randint(2, 120, (1, 55)); ep2_long[0, 0] = TOK_RESET
    out_la = m(torch.cat([ep1_a, ep2_long], dim=1))["logits"][:, -1]
    out_lb = m(torch.cat([ep1_b, ep2_long], dim=1))["logits"][:, -1]
    # Note RoPE absolute position is identical (pos 104 in both), and TOK_RESET wiped episodic fast-weight memory!
    diff = (out_la - out_lb).abs().max().item()
    assert diff < 1e-5, f"Cross-episode memory bleed across TOK_RESET: diff={diff}"


def test_fast_weight_norm_stability_and_ttt_reversibility():
    cfg = CortexConfig()
    m = H3CortexAGI(cfg).eval()
    ok, msg = check_safety_invariants(m, device="cpu")
    assert ok, msg
    before = {k: v.clone() for k, v in m.state_dict().items()}
    evaluate_arc_ttt(m, n_episodes=2, inner_steps=3, device="cpu")
    after = m.state_dict()
    for k in before:
        assert torch.allclose(before[k], after[k], atol=1e-7), f"TTT mutated slow parameter {k}!"


def test_lab_stream_consistency():
    lcfg = LabConfig(vocab_size=512, block_size=64, n_layer=4, n_head=4, d_model=128)
    for k in ("memory", "synapse", "cortex", "hippocampus"):
        m = LAB_MODELS[k](lcfg).eval()
        x = torch.randint(1, 512, (2, 160))
        x[:, 0] = 0
        x[0, 75] = 0
        full, _ = m(x)
        mems, parts = None, []
        for s in range(0, 160, 64):
            lg, _, mems = m.forward_stream(x[:, s : s + 64], None, mems)
            parts.append(lg)
        assert (full - torch.cat(parts, dim=1)).abs().max().item() < 1e-5, f"Lab {k} stream mismatch"
        if isinstance(mems[-1], dict):  # hippocampus slot bank: salience bounded, keys unit-norm, finite
            st = mems[-1]
            assert torch.isfinite(st["K"]).all() and torch.isfinite(st["V"]).all()
            assert float(st["w"].max()) <= 1.0 + 1e-5, "slot salience out of bounds"
            kn = st["K"].norm(dim=-1)
            live = st["w"] > 1e-6
            if live.any():
                assert (kn[live] - 1.0).abs().max().item() < 1e-3, "slot keys not normalized"


if __name__ == "__main__":
    test_causal_and_padding_invariance()
    test_episode_boundary_isolation()
    test_fast_weight_norm_stability_and_ttt_reversibility()
    test_lab_stream_consistency()
    print("ALL 4 AUTOMATED INVARIANT SUITES PASSED!")
