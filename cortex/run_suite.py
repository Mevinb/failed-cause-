"""Master Experimental Harness for Cortex-AGI (H0 vs H1 vs H2 vs H3 + TTT + Calibration + Planning + Continual)."""
import argparse
import inspect
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortex.arch import ARCH_REGISTRY, CortexConfig
from cortex.continual import run_continual_experiment
from cortex.planning import collect_world_model_dataset, evaluate_planning, train_world_model
from cortex.tasks import (
    ARC_OOD_OPS,
    TASK_ARC,
    TOK_RESET,
    TRAIN_SAMPLERS,
    _encode_grid_pair,
    collate_episodes,
    sample_arc_grid,
    sample_causal,
    sample_comp_math,
    sample_episodic,
    sample_multimodal,
    sample_s5,
)


def compute_ece_and_auroc(confs: list[float], corrects: list[int], n_bins: int = 10) -> tuple[float, float]:
    c_arr = np.clip(np.array(confs, dtype=np.float64), 0.0, 1.0)
    y_arr = np.array(corrects, dtype=np.float64)
    N = len(c_arr)
    if N == 0:
        return 0.0, 0.5
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (c_arr >= b0) & (c_arr <= b1 if b1 == 1.0 else c_arr < b1)
        if m.any():
            ece += (m.sum() / N) * abs(y_arr[m].mean() - c_arr[m].mean())

    # Mann-Whitney U AUROC (probability that a correct prediction has higher confidence than an incorrect one)
    pos = c_arr[y_arr == 1]
    neg = c_arr[y_arr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return round(float(ece), 4), 0.5
    diff = pos[:, None] - neg[None, :]
    auroc = float((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg))
    return round(float(ece), 4), round(auroc, 4)


def episode_balanced_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    tok_ce = F.cross_entropy(logits.reshape(B * T, V).float(), targets.reshape(B * T), ignore_index=-100, reduction="none").view(B, T)
    mask = (targets != -100).float()
    per_ep = (tok_ce * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return per_ep.mean()


def train_multitask_model(
    arch_name: str,
    cfg: CortexConfig,
    steps: int = 3600,
    batch_size: int = 36,
    lr: float = 7.5e-4,
    seed: int = 0,
    device: str = "cuda",
) -> tuple[torch.nn.Module, list[dict]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = ARCH_REGISTRY[arch_name](cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.04)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))

    history = []
    t0 = time.time()
    for step in range(steps):
        # Cosine LR schedule with 150-step warmup
        if step < 150:
            cur_lr = lr * (step + 1) / 150
        else:
            p = (step - 150) / max(1, steps - 150)
            cur_lr = lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * p)))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        # Balanced multi-task batch (6 episodes per each of the 6 core tasks)
        per_task = max(1, batch_size // len(TRAIN_SAMPLERS))
        eps = [fn(rng) for _, fn in TRAIN_SAMPLERS for _ in range(per_task)]
        b = collate_episodes(eps, device=device)

        # Randomize recurrence depth during training (Geiping et al. 2025) for test-time depth generalization
        k_train = rng.choice([2, 3, 4, 5, 6]) if cfg.ponder_steps >= 4 else cfg.ponder_steps

        with autocast:
            out = model(b["x"], b["rows"], b["cols"], b["modality"], ponder_steps=k_train)
            step_logits = out["step_logits"]
            conf_logits = out["conf_logits"]

            # Per-episode balanced deeply-supervised task loss across latent iterations
            weights = [k + 1 for k in range(len(step_logits))]
            task_loss = sum(w * episode_balanced_ce(lg, b["y"]) for w, lg in zip(weights, step_logits)) / sum(weights)

            # Epistemic Verifier training (Subsystem 8): predict target token correctness at each step k
            mask = b["y"] != -100
            ver_loss = torch.zeros((), device=device)
            if len(conf_logits) > 1 or hasattr(model, "reasoner"):
                for lg, cl in zip(step_logits, conf_logits):
                    with torch.no_grad():
                        tok_ok = (lg.argmax(dim=-1) == b["y"]).float()
                    ver_loss = ver_loss + F.binary_cross_entropy_with_logits(cl[mask].float(), tok_ok[mask])
                ver_loss = ver_loss / len(conf_logits)

            loss = task_loss + 0.25 * ver_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 400 == 0 or (step + 1) == steps:
            row = {
                "step": step + 1,
                "task_loss": round(float(task_loss.item()), 4),
                "ver_loss": round(float(ver_loss.item()), 4),
                "elapsed_s": round(time.time() - t0, 1),
            }
            history.append(row)
            print(f"[{arch_name}] step {step+1}/{steps} task_loss={row['task_loss']:.4f} ver_loss={row['ver_loss']:.4f} ({row['elapsed_s']}s)", flush=True)

    return model, history


@torch.no_grad()
def eval_split(
    model: torch.nn.Module,
    sampler,
    ood: str | None,
    n_eval: int = 240,
    seed: int = 2026,
    ponder_steps: int | None = None,
    verifier_select: bool = False,
    device: str = "cuda",
) -> dict:
    model.eval()
    rng = random.Random(seed)
    eps = [sampler(rng, ood=ood) for _ in range(n_eval)]
    exact_hits, tok_hits, tok_total = 0, 0, 0
    confs, corrects = [], []

    for s in range(0, n_eval, 40):
        sub = eps[s : s + 40]
        b = collate_episodes(sub, device=device)
        out = model(b["x"], b["rows"], b["cols"], b["modality"], ponder_steps=ponder_steps)
        step_logits = out["step_logits"]
        conf_logits = out["conf_logits"]

        for i in range(len(sub)):
            mask = b["y"][i] != -100
            if verifier_select and len(step_logits) > 1:
                # Subsystem 11 (Self-Correction): joint sequence validity prod_m p_m = exp(sum_m log_sigmoid(cl))
                step_scores = [torch.exp(F.logsigmoid(cl[i][mask].float()).sum()).item() for cl in conf_logits]
                best_k = int(np.argmax(step_scores))
                lg = step_logits[best_k][i]
                conf = step_scores[best_k]
            else:
                lg = step_logits[-1][i]
                conf = torch.exp(F.logsigmoid(conf_logits[-1][i][mask].float()).sum()).item()

            pred = lg.argmax(dim=-1)[mask]
            tgt = b["y"][i][mask]
            ok_exact = int((pred == tgt).all().item())
            exact_hits += ok_exact
            tok_hits += int((pred == tgt).sum().item())
            tok_total += int(mask.sum().item())
            confs.append(conf)
            corrects.append(ok_exact)

    ece, auroc = compute_ece_and_auroc(confs, corrects)
    return {
        "exact_acc": round(exact_hits / n_eval, 4),
        "token_acc": round(tok_hits / max(1, tok_total), 4),
        "ece": ece,
        "auroc": auroc,
        "confs": confs,
        "corrects": corrects,
    }


def evaluate_arc_ttt(
    model: torch.nn.Module,
    n_episodes: int = 80,
    inner_steps: int = 12,
    inner_lr: float = 2e-2,
    seed: int = 8888,
    device: str = "cuda",
) -> dict:
    """Subsystem 7: Test-Time Training (TTT) on 3 support demo pairs of unseen compositional 2D ARC rules."""
    rng = random.Random(seed)
    eps = [sample_arc_grid(rng, ood="comp_ood", specific_op=ARC_OOD_OPS[i % len(ARC_OOD_OPS)]) for i in range(n_episodes)]

    # Freeze slow weights; adapt ONLY fast low-rank ttt_adapter (+ FiLM program on Gen-2) per episode, then restore!
    saved_adapter = {k: v.detach().clone() for k, v in model.ttt_adapter.state_dict().items()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.ttt_adapter.parameters():
        p.requires_grad_(True)
    supports_program = "program" in inspect.signature(model.forward).parameters

    zs_exact, zs_cell, ttt_exact, ttt_cell = 0, 0, 0, 0

    for ep in eps:
        model.ttt_adapter.reset_adapter()
        model.eval()

        # 1. Zero-shot prediction before test-time adaptation
        b_full = collate_episodes([ep], device=device)
        with torch.no_grad():
            lg_zs = model(b_full["x"], b_full["rows"], b_full["cols"], b_full["modality"])["logits"][0]
            mask = b_full["y"][0] != -100
            p_zs = lg_zs.argmax(dim=-1)[mask]
            t_zs = b_full["y"][0][mask]
            zs_exact += int((p_zs == t_zs).all().item())
            zs_cell += int((p_zs == t_zs).sum().item())

        # 2. Construct leave-one-out support episodes from ONLY the 3 support demo pairs (zero test leakage!)
        demos = ep["demos"]
        sup_eps = []
        for hold in range(3):
            ctx_demos = [demos[j] for j in range(3) if j != hold]
            target_in, target_out = demos[hold]
            seq, rows, cols = [TOK_RESET, TASK_ARC], [0, 0], [0, 0]
            for gi, go in ctx_demos:
                t, r, c = _encode_grid_pair(gi, go)
                seq.extend(t); rows.extend(r); cols.extend(c)
            t, r, c = _encode_grid_pair(target_in, target_out)
            pref = len(seq) + 1 + 16
            seq.extend(t); rows.extend(r); cols.extend(c)
            lm = [0] * len(seq)
            for pos in range(pref, pref + 16):
                lm[pos - 1] = 1
            sup_eps.append({"seq": seq, "rows": rows, "cols": cols, "loss_mask": lm, "modality": 1})

        b_sup = collate_episodes(sup_eps, device=device)
        prog = None
        params = list(model.ttt_adapter.parameters())
        if supports_program:
            prog = torch.zeros(1, model.cfg.d_model, device=device, requires_grad=True)
            params.append(prog)
        inner_opt = torch.optim.AdamW(params, lr=inner_lr, weight_decay=0.0)
        for _ in range(inner_steps):
            kw = {"program": prog.expand(b_sup["x"].shape[0], -1)} if supports_program else {}
            out_s = model(b_sup["x"], b_sup["rows"], b_sup["cols"], b_sup["modality"], **kw)
            loss_s = F.cross_entropy(out_s["logits"].flatten(0, 1), b_sup["y"].flatten(), ignore_index=-100)
            inner_opt.zero_grad(set_to_none=True)
            loss_s.backward()
            inner_opt.step()

        # 3. Evaluate on the unseen test query grid using adapted fast weights (+ induced program)
        with torch.no_grad():
            kw = {"program": prog} if supports_program else {}
            lg_ttt = model(b_full["x"], b_full["rows"], b_full["cols"], b_full["modality"], **kw)["logits"][0]
            p_ttt = lg_ttt.argmax(dim=-1)[mask]
            ttt_exact += int((p_ttt == t_zs).all().item())
            ttt_cell += int((p_ttt == t_zs).sum().item())

    model.ttt_adapter.load_state_dict(saved_adapter)
    for p in model.parameters():
        p.requires_grad_(True)

    return {
        "zeroshot_grid_exact": round(zs_exact / n_episodes, 4),
        "zeroshot_cell_acc": round(zs_cell / (n_episodes * 16), 4),
        "ttt_grid_exact": round(ttt_exact / n_episodes, 4),
        "ttt_cell_acc": round(ttt_cell / (n_episodes * 16), 4),
    }


def evaluate_all_axes(model: torch.nn.Module, arch_name: str, device: str = "cuda") -> dict:
    eval_matrix = [
        ("episodic_id_gap64", sample_episodic, "id"),
        ("episodic_ood_gap160", sample_episodic, "long160"),
        ("episodic_ood_gap320", sample_episodic, "long320"),
        ("s5_id_len6", sample_s5, "id"),
        ("s5_ood_len10", sample_s5, "len10"),
        ("s5_ood_len13", sample_s5, "len13"),
        ("math_id_depth2", sample_comp_math, "id"),
        ("math_ood_depth3", sample_comp_math, "depth3"),
        ("math_ood_depth4", sample_comp_math, "depth4"),
        ("arc_id_3demo", sample_arc_grid, "id"),
        ("arc_ood_compositional", sample_arc_grid, "comp_ood"),
        ("causal_id_mixed", sample_causal, "id"),
        ("causal_ood_scm", sample_causal, "ood_scm"),
        ("causal_ood_do_interventional", sample_causal, "ood_do_only"),
        ("multimodal_id_1hop", sample_multimodal, "id"),
        ("multimodal_ood_2hop", sample_multimodal, "hops2"),
    ]
    scores = {}
    all_confs, all_corrs = [], []
    for key, fn, split in eval_matrix:
        r = eval_split(model, fn, split, device=device)
        all_confs.extend(r.pop("confs"))
        all_corrs.extend(r.pop("corrects"))
        scores[key] = r

    ece_global, auroc_global = compute_ece_and_auroc(all_confs, all_corrs)
    scores["metacognitive_global"] = {"ece": ece_global, "error_detect_auroc": auroc_global}

    # Test-time compute scaling curve (K = 1..8, +12 for Gen-2 sinusoidal depth) on hard OOD reasoning tasks
    if arch_name in ("H2_latent_ponder", "H3_cortex_agi", "H4_cortex_v2"):
        ponder_curve = {}
        k_list = [1, 2, 4, 6, 8] + ([12, 16] if arch_name == "H4_cortex_v2" else [])
        for K in k_list:
            m_d3 = eval_split(model, sample_comp_math, "depth3", ponder_steps=K, device=device)["exact_acc"]
            s5_10 = eval_split(model, sample_s5, "len10", ponder_steps=K, device=device)["exact_acc"]
            mm_2h = eval_split(model, sample_multimodal, "hops2", ponder_steps=K, device=device)["exact_acc"]
            ponder_curve[f"K={K}"] = {
                "math_ood_d3": m_d3,
                "s5_ood_len10": s5_10,
                "multimodal_ood_2hop": mm_2h,
                "mean_reasoning_ood": round((m_d3 + s5_10 + mm_2h) / 3, 4),
            }
        # Verifier-guided self-correction at K=8 vs raw K=8 and K=1
        ver_m = eval_split(model, sample_comp_math, "depth3", ponder_steps=8, verifier_select=True, device=device)["exact_acc"]
        ver_s5 = eval_split(model, sample_s5, "len10", ponder_steps=8, verifier_select=True, device=device)["exact_acc"]
        ver_mm = eval_split(model, sample_multimodal, "hops2", ponder_steps=8, verifier_select=True, device=device)["exact_acc"]
        ponder_curve["K=8_verifier_selected"] = {
            "math_ood_d3": ver_m,
            "s5_ood_len10": ver_s5,
            "multimodal_ood_2hop": ver_mm,
            "mean_reasoning_ood": round((ver_m + ver_s5 + ver_mm) / 3, 4),
        }
        scores["test_time_compute_scaling"] = ponder_curve

    # Aggregate ID and OOD composite scores
    id_keys = [k for k, _, _ in eval_matrix if "_id_" in k]
    ood_keys = [k for k, _, _ in eval_matrix if "_ood_" in k]
    scores["composite_id_acc"] = round(float(np.mean([scores[k]["exact_acc"] for k in id_keys])), 4)
    scores["composite_ood_acc"] = round(float(np.mean([scores[k]["exact_acc"] for k in ood_keys])), 4)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--batch", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "cortex_runs/benchmark_results.json"))
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CortexConfig()
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    full_report = {"config": asdict(cfg), "steps": a.steps, "seed": a.seed, "architectures": {}}
    wm_dataset = collect_world_model_dataset(n_envs=750, seed_start=10000)

    for arch_name in ["H0_transformer", "H1_synapse_mem", "H2_latent_ponder", "H3_cortex_agi", "H4_cortex_v2"]:
        print(f"\n================ TRAINING & EVALUATING {arch_name} ================", flush=True)
        model, hist = train_multitask_model(arch_name, cfg, steps=a.steps, batch_size=a.batch, seed=a.seed, device=dev)
        n_params = sum(p.numel() for p in model.parameters())
        scores = evaluate_all_axes(model, arch_name, device=dev)
        ttt_res = evaluate_arc_ttt(model, n_episodes=80, device=dev)

        # Save checkpoint
        ckpt_path = out_path.parent / f"{arch_name}.pt"
        torch.save({"arch": arch_name, "cfg": asdict(cfg), "model": model.state_dict()}, ckpt_path)

        # Train & evaluate World-Model MPC Lookahead on the unified architectures
        plan_res = None
        if arch_name in ("H3_cortex_agi", "H4_cortex_v2"):
            print(f"[{arch_name}] Training LatentWorldModel & running Key-Door-Trap MPC planning...", flush=True)
            train_world_model(model, wm_dataset, steps=900, device=dev)
            plan_res = evaluate_planning(model, n_maps=80, device=dev)
            print(f"[{arch_name}] Planning: {plan_res}", flush=True)

        full_report["architectures"][arch_name] = {
            "params": n_params,
            "train_history": hist,
            "benchmarks": scores,
            "arc_ood_ttt": ttt_res,
            "planning": plan_res,
        }
        print(
            f"[{arch_name}] ID={scores['composite_id_acc']:.4f} | OOD={scores['composite_ood_acc']:.4f} | "
            f"ECE={scores['metacognitive_global']['ece']:.4f} | AUROC={scores['metacognitive_global']['error_detect_auroc']:.4f} | "
            f"ARC-OOD cell ZS={ttt_res['zeroshot_cell_acc']:.4f} -> TTT={ttt_res['ttt_cell_acc']:.4f}",
            flush=True,
        )
        out_path.write_text(json.dumps(full_report, indent=2))

    print("\n================ RUNNING 5-PHASE CONTINUAL LEARNING EXPERIMENT ================", flush=True)
    from cortex.arch import H3CortexAGI, H4CortexV2
    cl_res = {}
    for cl_name, cl_cls in [("H3_cortex_agi", H3CortexAGI), ("H4_cortex_v2", H4CortexV2)]:
        cl_res[cl_name] = run_continual_experiment(steps_per_phase=420, batch_size=32, device=dev, model_cls=cl_cls)
    full_report["continual_5phase"] = cl_res
    out_path.write_text(json.dumps(full_report, indent=2))
    for arch_name, res in cl_res.items():
        for reg, vals in res.items():
            print(
                f"[Continual:{arch_name}:{reg:<18}] FAA={vals['final_avg_acc']:.4f} | BWT={vals['backward_transfer']:+.4f} | "
                f"P5_Forward={vals['phase5_forward_acc']:.4f} | Dormant={vals['dormant_unit_frac']:.4f}",
                flush=True,
            )
    print("\nSUITE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
