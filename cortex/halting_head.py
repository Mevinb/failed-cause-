"""Exp #031 — Trained halting head (Wall-3 follow-up to Exp #030).
Exp #030 showed: dz-halting saves compute but oracle-K is far better; the gap is
that dz signals CONVERGENCE not CORRECTNESS, and the learned verifier fails OOD.
Here a small MLP head is trained on FROZEN H4 features (conf, dz, step index,
entropy) to predict per-step exact-correctness on the ID distribution, then
evaluated OOD: does argmax-p_halt recover more of the oracle-K gap than
dz-tau or verifier-select? Features are scalar trajectories, so OOD transfer
is the falsifiable question.
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


def collect(model, sampler, ood, n_eval, seed, device):
    """Per-episode per-step features + correctness label."""
    rng = random.Random(seed)
    eps = [sampler(rng, ood=ood) for _ in range(n_eval)]
    feats, labels = [], []
    for s in range(0, n_eval, 40):
        sub = eps[s : s + 40]
        b = collate_episodes(sub, device=device)
        with torch.no_grad():
            out = model(b["x"], b["rows"], b["cols"], b["modality"], ponder_steps=K_MAX)
        for i in range(len(sub)):
            mask = b["y"][i] != -100
            tgt = b["y"][i][mask]
            prev_dz, prev_conf = 0.0, 0.0
            for k in range(min(K_MAX, len(out["step_logits"]))):
                lg = out["step_logits"][k][i][mask].float()
                pred = lg.argmax(dim=-1)
                correct = float((pred == tgt).all().item())
                p = lg.softmax(dim=-1)
                ent = float(-(p * p.clamp_min(1e-9).log()).sum(-1).mean().item())
                conf = float(torch.exp(F.logsigmoid(out["conf_logits"][k][i][mask].float()).sum()).item())
                dz = out["dz"][k][i][mask].float().mean().item()
                feats.append([dz, conf, k / K_MAX, ent, dz - prev_dz, conf - prev_conf])
                labels.append(correct)
                prev_dz, prev_conf = dz, conf
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.float32)


class HaltHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def eval_policy(feats, labels, probs, rule):
    """feats/labels/probs: [E, K] arrays. Returns (acc, mean_K)."""
    E, K = labels.shape
    if rule == "oracle":
        stops = labels.argmax(axis=1)
        return float(labels.max(axis=1).mean()), float(stops.mean() + 1)
    if rule == "halt_argmax":
        stops = probs.argmax(axis=1)
    elif rule.startswith("halt_thr"):
        thr = float(rule[8:])
        stops = np.full(E, K - 1)
        for k in range(K):
            take = (probs[:, k] > thr) & (stops == K - 1)
            stops[take] = k
    elif rule.startswith("dz"):
        tau = float(rule[2:])
        stops = np.full(E, K - 1)
        for k in range(K):
            take = (feats[:, k, 0] < tau) & (stops == K - 1)
            stops[take] = k
    elif rule == "conf_argmax":
        stops = feats[:, :, 1].argmax(axis=1)
    idx = np.arange(E)
    return float(labels[idx, stops].mean()), float(stops.mean() + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="cortex_runs/H4_cortex_v2.pt")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=160)
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="cortex_runs/halting_head.json")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ARCH_REGISTRY[ck["arch"]](CortexConfig(**ck["cfg"])).to(a.device)
    model.load_state_dict(ck["model"])
    model.eval()

    # 1) train the head on ID-distribution features
    id_tasks = [(sample_comp_math, "id"), (sample_s5, "id"), (sample_multimodal, "id")]
    X_parts, y_parts = [], []
    for sampler, ood in id_tasks:
        f_, l_ = collect(model, sampler, ood, a.n_train // 3, a.seed, a.device)
        X_parts.append(f_); y_parts.append(l_)
    X = torch.tensor(np.concatenate(X_parts))
    y = torch.tensor(np.concatenate(y_parts))
    head = HaltHead()
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    pos_w = (1 - y.mean()).clamp_min(0.05) / y.mean().clamp_min(0.05)
    for it in range(600):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(X), y, pos_weight=pos_w)
        loss.backward(); opt.step()
    print(f"[halt_head] trained {it+1} iters, final loss {loss.item():.4f}, pos_rate {y.mean().item():.3f}")

    # 2) OOD eval per task
    ood_tasks = [("math_ood_depth3", sample_comp_math, "depth3"),
                 ("s5_ood_len10", sample_s5, "len10"),
                 ("multimodal_ood_2hop", sample_multimodal, "hops2")]
    report = {}
    for name, sampler, ood in ood_tasks:
        f_, l_ = collect(model, sampler, ood, a.n_eval, a.seed + 1, a.device)
        E = f_.shape[0] // K_MAX
        feats = f_.reshape(E, K_MAX, -1)
        labels = l_.reshape(E, K_MAX)
        with torch.no_grad():
            probs = torch.sigmoid(head(torch.tensor(feats))).numpy()
        res = {"perstep_acc": [float(labels[:, k].mean()) for k in range(K_MAX)]}
        for rule in ["oracle", "halt_argmax", "halt_thr0.5", "halt_thr0.7",
                     "dz0.5", "dz0.6", "conf_argmax"]:
            acc, mk = eval_policy(feats, labels, probs, rule)
            res[rule] = {"acc": acc, "mean_K": mk}
            print(f"    [{name}] {rule:14s} acc={acc:.3f} meanK={mk:.1f}")
        report[name] = res
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
