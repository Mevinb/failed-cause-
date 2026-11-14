"""Exp #030 — Adaptive halting probe (Wall-3): does the `dz` contraction signal
support early-exit? Runs the trained H4 reasoner to K=16 and compares:
  - fixed-K baselines (K=4, 8, 16)
  - oracle-K (per-episode best step — upper bound)
  - dz-threshold halting: first step where mean masked dz < tau
  - conf-threshold halting: first step where sequence confidence > c
  - verifier-select (existing post-hoc argmax over all K — reference)
Reports accuracy vs mean-compute (steps actually used).
Eval-only on an existing checkpoint; no training changes.
"""
import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cortex.arch import ARCH_REGISTRY, CortexConfig
from cortex.tasks import collate_episodes, sample_comp_math, sample_multimodal, sample_s5

K_MAX = 16


def collect_stepwise(model, sampler, ood, n_eval, seed, device):
    """Per-episode, per-ponder-step: exact hit, mean masked dz, sequence confidence."""
    rng = random.Random(seed)
    eps = [sampler(rng, ood=ood) for _ in range(n_eval)]
    hits = np.zeros((n_eval, K_MAX))          # [E, K]
    dzs = np.zeros((n_eval, K_MAX))
    confs = np.zeros((n_eval, K_MAX))
    for s in range(0, n_eval, 40):
        sub = eps[s : s + 40]
        b = collate_episodes(sub, device=device)
        with torch.no_grad():
            out = model(b["x"], b["rows"], b["cols"], b["modality"], ponder_steps=K_MAX)
        for i in range(len(sub)):
            mask = b["y"][i] != -100
            tgt = b["y"][i][mask]
            for k in range(min(K_MAX, len(out["step_logits"]))):
                pred = out["step_logits"][k][i].argmax(dim=-1)[mask]
                hits[s + i, k] = float((pred == tgt).all().item())
                dzs[s + i, k] = out["dz"][k][i][mask].float().mean().item()
                confs[s + i, k] = float(torch.exp(F.logsigmoid(out["conf_logits"][k][i][mask].float()).sum()).item())
    return {"hits": hits, "dzs": dzs, "confs": confs}


def analyze(data: dict) -> dict:
    hits, dzs, confs = data["hits"], data["dzs"], data["confs"]
    E, K = hits.shape
    res = {}
    # fixed-K baselines (use step K-1's output)
    for kk in (4, 8, 16):
        res[f"fixed_K{kk}"] = {"acc": float(hits[:, kk - 1].mean()), "mean_K": float(kk)}
    # oracle-K: per-episode best step
    res["oracle_K"] = {"acc": float(hits.max(axis=1).mean()), "mean_K": float((hits.argmax(axis=1) + 1).mean())}
    # verifier-select: argmax sequence conf across K (post-hoc — sees all K)
    vs = confs.argmax(axis=1)
    res["verifier_select"] = {"acc": float(hits[np.arange(E), vs].mean()), "mean_K": float(K)}
    # dz-threshold halting: first k where dz < tau (fallback to K)
    for tau in np.quantile(dzs.ravel(), [0.05, 0.15, 0.3, 0.5, 0.7]):
        stops = np.full(E, K - 1)
        for k in range(K):
            take = (dzs[:, k] < tau) & (stops == K - 1)
            stops[take] = k
        res[f"dz_tau{tau:.4f}"] = {"acc": float(hits[np.arange(E), stops].mean()), "mean_K": float(stops.mean() + 1)}
    # conf-threshold halting: first k where conf > c
    for c in np.quantile(confs.ravel(), [0.3, 0.5, 0.7, 0.9]):
        stops = np.full(E, K - 1)
        for k in range(K):
            take = (confs[:, k] > c) & (stops == K - 1)
            stops[take] = k
        res[f"conf_c{c:.3f}"] = {"acc": float(hits[np.arange(E), stops].mean()), "mean_K": float(stops.mean() + 1)}
    res["_perstep_acc"] = [float(hits[:, k].mean()) for k in range(K)]
    res["_perstep_dz"] = [float(dzs[:, k].mean()) for k in range(K)]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="cortex_runs/H4_cortex_v2.pt")
    ap.add_argument("--n-eval", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="cortex_runs/halting_probe.json")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ARCH_REGISTRY[ck["arch"]](CortexConfig(**ck["cfg"])).to(a.device)
    model.load_state_dict(ck["model"])
    model.eval()

    tasks = [
        ("math_ood_depth3", sample_comp_math, "depth3"),
        ("s5_ood_len10", sample_s5, "len10"),
        ("multimodal_ood_2hop", sample_multimodal, "hops2"),
    ]
    report = {}
    for name, sampler, ood in tasks:
        data = collect_stepwise(model, sampler, ood, a.n_eval, a.seed, a.device)
        report[name] = analyze(data)
        print(f"[{name}] perstep_acc={np.round(report[name]['_perstep_acc'], 3)}")
        print(f"[{name}] perstep_dz ={np.round(report[name]['_perstep_dz'], 4)}")
        for k_, v in report[name].items():
            if not k_.startswith("_"):
                print(f"    {k_:22s} acc={v['acc']:.3f} meanK={v['mean_K']:.1f}")
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
