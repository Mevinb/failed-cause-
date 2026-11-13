"""Post-Grokking Controlled Experiment: Algebraic State Tracking (S_5), Causal do-Calculus & Compositional Math.

Tests why 21.6k multi-task samples in 6-way padded training stayed pre-grokking on algebraic tasks,
and compares iso-parameter H0, H1, H2, and H3 at 4,500 steps x batch 120 (540k episodes, ~45s/model)
on ID mastery, strict OOD length/depth/SCM generalization, Test-Time Compute Scaling (K=1..8), and
Verifier-Guided Self-Correction.
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortex.arch import ARCH_REGISTRY, CortexConfig
from cortex.run_suite import episode_balanced_ce, eval_split
from cortex.tasks import collate_episodes, sample_causal, sample_comp_math, sample_s5

GROK_SAMPLERS = [
    ("s5", sample_s5),
    ("math", sample_comp_math),
    ("causal", sample_causal),
]


def run_grok_experiment(steps: int = 4500, batch_size: int = 120, lr: float = 1.0e-3, seed: int = 0, device: str = "cuda"):
    cfg = CortexConfig(max_seq_len=128, local_window=48, micro_chunk=8)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))
    results = {"steps": steps, "batch_size": batch_size, "episodes_seen": steps * batch_size, "models": {}}

    for arch_name in ["H0_transformer", "H1_synapse_mem", "H2_latent_ponder", "H3_cortex_agi"]:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model = ARCH_REGISTRY[arch_name](cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
        t0 = time.time()

        for step in range(steps):
            if step < 200:
                cur_lr = lr * (step + 1) / 200
            else:
                p = (step - 200) / max(1, steps - 200)
                cur_lr = lr * (0.08 + 0.46 * (1.0 + math.cos(math.pi * p)))
            for g in opt.param_groups:
                g["lr"] = cur_lr

            per_t = batch_size // len(GROK_SAMPLERS)
            eps = [fn(rng) for _, fn in GROK_SAMPLERS for _ in range(per_t)]
            b = collate_episodes(eps, device=device)
            k_tr = rng.choice([2, 3, 4, 5, 6]) if hasattr(model, "reasoner") else None

            with autocast:
                out = model(b["x"], b["rows"], b["cols"], b["modality"], ponder_steps=k_tr)
                step_lgs = out["step_logits"]
                conf_lgs = out["conf_logits"]
                weights = [k + 1 for k in range(len(step_lgs))]
                t_loss = sum(w * episode_balanced_ce(lg, b["y"]) for w, lg in zip(weights, step_lgs)) / sum(weights)

                mask = b["y"] != -100
                v_loss = torch.zeros((), device=device)
                if hasattr(model, "reasoner"):
                    for lg, cl in zip(step_lgs, conf_lgs):
                        with torch.no_grad():
                            ok = (lg.argmax(dim=-1) == b["y"]).float()
                        v_loss = v_loss + F.binary_cross_entropy_with_logits(cl[mask].float(), ok[mask])
                    v_loss = v_loss / len(conf_lgs)
                loss = t_loss + 0.25 * v_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if (step + 1) % 900 == 0:
                print(f"[Grok:{arch_name}] step {step+1}/{steps} loss={t_loss.item():.4f} ({time.time()-t0:.1f}s)", flush=True)

        eval_splits = [
            ("s5_id_len6", sample_s5, "id"),
            ("s5_ood_len10", sample_s5, "len10"),
            ("s5_ood_len13", sample_s5, "len13"),
            ("math_id_depth2", sample_comp_math, "id"),
            ("math_ood_depth3", sample_comp_math, "depth3"),
            ("math_ood_depth4", sample_comp_math, "depth4"),
            ("causal_id_mixed", sample_causal, "id"),
            ("causal_ood_scm", sample_causal, "ood_scm"),
            ("causal_ood_do_only", sample_causal, "ood_do_only"),
        ]
        sc = {}
        for k, fn, sp in eval_splits:
            r = eval_split(model, fn, sp, n_eval=320, seed=4040, device=device)
            sc[k] = {"acc": r["exact_acc"], "ece": r["ece"], "auroc": r["auroc"]}

        if hasattr(model, "reasoner"):
            k_curve = {}
            for K in [1, 2, 3, 4, 6, 8]:
                a_s5 = eval_split(model, sample_s5, "len10", n_eval=320, seed=4040, ponder_steps=K, device=device)["exact_acc"]
                a_ma = eval_split(model, sample_comp_math, "depth3", n_eval=320, seed=4040, ponder_steps=K, device=device)["exact_acc"]
                a_ca = eval_split(model, sample_causal, "ood_do_only", n_eval=320, seed=4040, ponder_steps=K, device=device)["exact_acc"]
                k_curve[f"K={K}"] = {"s5_len10": a_s5, "math_d3": a_ma, "causal_do": a_ca, "mean": round((a_s5 + a_ma + a_ca) / 3, 4)}
            vs_s5 = eval_split(model, sample_s5, "len10", n_eval=320, seed=4040, ponder_steps=8, verifier_select=True, device=device)["exact_acc"]
            vs_ma = eval_split(model, sample_comp_math, "depth3", n_eval=320, seed=4040, ponder_steps=8, verifier_select=True, device=device)["exact_acc"]
            vs_ca = eval_split(model, sample_causal, "ood_do_only", n_eval=320, seed=4040, ponder_steps=8, verifier_select=True, device=device)["exact_acc"]
            k_curve["K=8_verifier_selected"] = {"s5_len10": vs_s5, "math_d3": vs_ma, "causal_do": vs_ca, "mean": round((vs_s5 + vs_ma + vs_ca) / 3, 4)}
            sc["compute_scaling"] = k_curve

        id_mean = float(np.mean([sc[k]["acc"] for k, _, _ in eval_splits if "_id_" in k]))
        ood_mean = float(np.mean([sc[k]["acc"] for k, _, _ in eval_splits if "_ood_" in k]))
        sc["id_mean"] = round(id_mean, 4)
        sc["ood_mean"] = round(ood_mean, 4)
        results["models"][arch_name] = sc
        print(f"[Grok:{arch_name}] ID={id_mean:.4f} | OOD={ood_mean:.4f} | S5_ID={sc['s5_id_len6']['acc']:.4f} S5_OOD10={sc['s5_ood_len10']['acc']:.4f} | Math_ID={sc['math_id_depth2']['acc']:.4f} | Causal_DO={sc['causal_ood_do_only']['acc']:.4f}", flush=True)
        if "compute_scaling" in sc:
            print(f"   Scaling: K=1:{sc['compute_scaling']['K=1']['mean']:.4f} -> K=2:{sc['compute_scaling']['K=2']['mean']:.4f} -> K=4:{sc['compute_scaling']['K=4']['mean']:.4f} -> K=6:{sc['compute_scaling']['K=6']['mean']:.4f} -> K=8_ver:{sc['compute_scaling']['K=8_verifier_selected']['mean']:.4f}", flush=True)

    out_p = ROOT / "cortex_runs/reasoning_grok_results.json"
    out_p.write_text(json.dumps(results, indent=2))
    print("GROK_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    run_grok_experiment()
