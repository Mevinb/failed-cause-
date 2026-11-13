"""Axis 8: Continual 5-Phase Lifelong Learning & Dual Stability-Plasticity Engine."""
import copy
import random

import torch
import torch.nn.functional as F

from cortex.arch import CortexConfig, H3CortexAGI
from cortex.tasks import (
    collate_episodes,
    sample_causal,
    sample_comp_math,
    sample_episodic,
    sample_multimodal,
    sample_s5,
)

PHASES = [
    ("P1_episodic", sample_episodic),
    ("P2_s5_state", sample_s5),
    ("P3_comp_math", sample_comp_math),
    ("P4_causal_do", sample_causal),
    ("P5_multimodal", sample_multimodal),
]


class SurpriseReplayBuffer:
    """Stores highest-loss / highest-surprise episodic exemplars per task phase."""

    def __init__(self, per_phase_cap: int = 48):
        self.per_phase_cap = per_phase_cap
        self.store: dict[str, list[tuple[float, dict]]] = {}

    def add_batch(self, phase_name: str, episodes: list[dict], scores: list[float]):
        bucket = self.store.setdefault(phase_name, [])
        for ep, sc in zip(episodes, scores):
            bucket.append((float(sc), ep))
        # Keep top half high-surprise exemplars + random diversity half to prevent outlier overfitting
        if len(bucket) > self.per_phase_cap:
            bucket.sort(key=lambda x: x[0], reverse=True)
            half = self.per_phase_cap // 2
            top = bucket[:half]
            rest = bucket[half:]
            random.Random(len(bucket)).shuffle(rest)
            self.store[phase_name] = top + rest[: self.per_phase_cap - half]

    def sample(self, rng: random.Random, k: int) -> list[dict]:
        all_eps = [ep for bucket in self.store.values() for _, ep in bucket]
        if not all_eps:
            return []
        return [rng.choice(all_eps) for _ in range(k)]


@torch.no_grad()
def shrink_and_perturb_plus_regen(model: torch.nn.Module, init_state: dict, shrink: float = 0.992, noise_scale: float = 0.008):
    """Dohare et al. (Nature 2024) plasticity preservation: pull weights slightly toward fresh init diversity."""
    for name, p in model.named_parameters():
        if p.dim() >= 2 and name in init_state and not name.startswith("perc."):
            p.mul_(shrink).add_(init_state[name], alpha=noise_scale)


@torch.no_grad()
def measure_dormant_fraction(model: H3CortexAGI, episodes: list[dict], device: str = "cuda", tau: float = 0.02) -> float:
    """Fraction of SwiGLU hidden units whose mean normalized activation falls below threshold tau."""
    acts = []
    hooks = []

    def make_hook():
        def fn(_mod, inp, _out):
            # inp[0] into down projection is post-SwiGLU hidden state (B, T, H)
            acts.append(inp[0].detach().abs().mean(dim=(0, 1)))
        return fn

    for m in model.modules():
        if hasattr(m, "down") and isinstance(m.down, torch.nn.Linear):
            hooks.append(m.down.register_forward_hook(make_hook()))
    try:
        batch = collate_episodes(episodes[:32], device=device)
        model(batch["x"], batch["rows"], batch["cols"], batch["modality"])
    finally:
        for h in hooks:
            h.remove()
    if not acts:
        return 0.0
    dormant, total = 0, 0
    for a in acts:
        norm_a = a / (a.mean() + 1e-8)
        dormant += int((norm_a < tau).sum().item())
        total += a.numel()
    return round(dormant / max(1, total), 4)


@torch.no_grad()
def eval_phase_acc(model: torch.nn.Module, sampler, n_eval: int = 160, seed: int = 9000, device: str = "cuda") -> float:
    model.eval()
    rng = random.Random(seed)
    eps = [sampler(rng, ood="id") for _ in range(n_eval)]
    correct = 0
    for s in range(0, n_eval, 32):
        sub = eps[s : s + 32]
        b = collate_episodes(sub, device=device)
        logits = model(b["x"], b["rows"], b["cols"], b["modality"])["logits"]
        preds = logits.argmax(dim=-1)
        for i in range(len(sub)):
            mask = b["y"][i] != -100
            correct += int((preds[i][mask] == b["y"][i][mask]).all().item())
    return correct / n_eval


def run_continual_experiment(steps_per_phase: int = 320, batch_size: int = 32, device: str = "cuda", model_cls=None) -> dict:
    """Evaluate 4 continual learning regimes across 5 sequential cognitive task phases."""
    if model_cls is None:
        model_cls = H3CortexAGI
    cfg = CortexConfig()
    regimes = ["naive_adamw", "high_l2_rigid", "uniform_replay", "cortex_plasticity"]
    summary = {}

    # Fixed validation probe episodes for dormant-unit tracking
    probe_rng = random.Random(777)
    probe_eps = [fn(probe_rng, ood="id") for _, fn in PHASES for _ in range(8)]

    for regime in regimes:
        torch.manual_seed(0)
        model = model_cls(cfg).to(device)
        # Fresh random reference distribution for shrink-and-perturb plasticity regeneration
        torch.manual_seed(101)
        fresh_ref = {k: v.detach().clone() for k, v in model_cls(cfg).to(device).named_parameters()}

        wd = 0.8 if regime == "high_l2_rigid" else 0.05
        lr = 2e-4 if regime == "high_l2_rigid" else 9e-4
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        replay = SurpriseReplayBuffer(per_phase_cap=48)

        # acc_matrix[p_after][p_eval]
        acc_matrix = []
        rng = random.Random(1234)

        for p_idx, (p_name, sampler) in enumerate(PHASES):
            model.train()
            if regime == "cortex_plasticity" and p_idx > 0:
                shrink_and_perturb_plus_regen(model, fresh_ref, shrink=0.985, noise_scale=0.015)

            for step in range(steps_per_phase):
                if regime in ("uniform_replay", "cortex_plasticity") and p_idx > 0:
                    n_rep = batch_size // 3
                    rep_eps = replay.sample(rng, n_rep)
                    cur_eps = [sampler(rng) for _ in range(batch_size - len(rep_eps))]
                    eps = cur_eps + rep_eps
                else:
                    cur_eps = [sampler(rng) for _ in range(batch_size)]
                    eps = cur_eps

                b = collate_episodes(eps, device=device)
                out = model(b["x"], b["rows"], b["cols"], b["modality"])
                # Episode-balanced deep supervision across latent iterations
                B_ep, T_ep, V_ep = out["logits"].shape
                mask_ep = (b["y"] != -100).float()
                denom_ep = mask_ep.sum(dim=1).clamp_min(1.0)
                losses = []
                for lg in out["step_logits"]:
                    t_ce = F.cross_entropy(lg.reshape(B_ep * T_ep, V_ep).float(), b["y"].reshape(B_ep * T_ep), ignore_index=-100, reduction="none").view(B_ep, T_ep)
                    losses.append(((t_ce * mask_ep).sum(dim=1) / denom_ep).mean())
                loss = sum((k + 1) * l for k, l in enumerate(losses)) / sum(range(1, len(losses) + 1))

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                # Populate replay buffer in final 20 steps of phase
                if regime in ("uniform_replay", "cortex_plasticity") and step >= steps_per_phase - 16:
                    with torch.no_grad():
                        per_tok = F.cross_entropy(
                            out["logits"][: len(cur_eps)].flatten(0, 1),
                            b["y"][: len(cur_eps)].flatten(),
                            ignore_index=-100,
                            reduction="none",
                        ).view(len(cur_eps), -1)
                        mask = (b["y"][: len(cur_eps)] != -100).float()
                        ep_loss = (per_tok * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                        scores = ep_loss.tolist() if regime == "cortex_plasticity" else [1.0] * len(cur_eps)
                    replay.add_batch(p_name, cur_eps, scores)

            row_accs = [round(eval_phase_acc(model, fn, device=device), 4) for _, fn in PHASES]
            acc_matrix.append(row_accs)

        # Compute standard Continual Learning metrics:
        # Final Average Accuracy (FAA), Backward Transfer (BWT), Phase-5 Forward Plasticity, Dormant Unit Fraction
        final_accs = acc_matrix[-1]
        faa = sum(final_accs) / len(PHASES)
        bwt = sum(acc_matrix[-1][i] - acc_matrix[i][i] for i in range(len(PHASES) - 1)) / (len(PHASES) - 1)
        p5_acc = acc_matrix[-1][-1]
        dormant = measure_dormant_fraction(model, probe_eps, device=device)

        summary[regime] = {
            "final_avg_acc": round(faa, 4),
            "backward_transfer": round(bwt, 4),
            "phase5_forward_acc": round(p5_acc, 4),
            "dormant_unit_frac": dormant,
            "final_per_phase": {PHASES[i][0]: final_accs[i] for i in range(len(PHASES))},
            "diagonal_peak": [acc_matrix[i][i] for i in range(len(PHASES))],
        }
    return summary
