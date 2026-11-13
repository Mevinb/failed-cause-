"""Darwin-Godel-Inspired Autonomous Hypothesis Generation, Ablation & Reversible Self-Improvement Loop.

Inspects multi-axis validation weaknesses of Cortex-AGI, formulates falsifiable architectural/learning
hypotheses, executes controlled trials under matched compute, enforces safety & calibration invariants,
records all successes AND failures in an append-only archive, and verifies the promoted champion on an
untouched held-out OOD test split.
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortex.arch import ARCH_REGISTRY, CortexConfig, H3CortexAGI
from cortex.run_suite import episode_balanced_ce, eval_split, evaluate_all_axes, evaluate_arc_ttt, train_multitask_model
from cortex.tasks import (
    sample_arc_grid,
    sample_causal,
    sample_comp_math,
    sample_episodic,
    sample_multimodal,
    sample_s5,
)


def validation_score(model: torch.nn.Module, ponder_eval: int | None = None, verifier_select: bool = False, device: str = "cuda") -> dict:
    """Evaluate candidate strictly on a dedicated Validation Seed (9999 != Test Seed 2026)."""
    splits = [
        ("ep_ood", sample_episodic, "long160"),
        ("s5_ood", sample_s5, "len10"),
        ("math_ood", sample_comp_math, "depth3"),
        ("arc_ood", sample_arc_grid, "comp_ood"),
        ("causal_do", sample_causal, "ood_do_only"),
        ("mm_2hop", sample_multimodal, "hops2"),
    ]
    accs, eces = {}, []
    for name, fn, sp in splits:
        r = eval_split(model, fn, sp, n_eval=160, seed=9999, ponder_steps=ponder_eval, verifier_select=verifier_select, device=device)
        accs[name] = r["exact_acc"]
        eces.append(r["ece"])
    mean_ood = float(np.mean(list(accs.values())))
    mean_ece = float(np.mean(eces))
    return {"per_axis": accs, "val_ood_mean": round(mean_ood, 4), "val_ece": round(mean_ece, 4)}


@torch.no_grad()
def check_safety_invariants(model: H3CortexAGI, device: str = "cuda") -> tuple[bool, str]:
    """Verify finite parameters, bounded fast-weight spectral/Frobenius norms over long random streams, and slow-weight integrity."""
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            return False, f"Non-finite parameter in {name}"
    # Stream 400 random non-reset tokens and verify episodic fast-weight matrix norm remains strictly bounded
    x = torch.randint(2, 200, (2, 400), device=device)
    emb = model.perc(x)
    reset_mask = x == 0
    max_norm = 0.0
    for b in getattr(model, "mem_blocks", []):
        emb, mem, _ = b(emb, model.rope, model.local_mask, reset_mask, None)
        max_norm = max(max_norm, float(mem.norm(dim=(-2, -1)).max().item()))
    if not np.isfinite(max_norm) or max_norm > 50.0:
        return False, f"Unbounded fast-weight memory norm: {max_norm:.2f}"
    return True, f"OK (max_mem_norm={max_norm:.3f})"


def generate_hypotheses_from_diagnostics(base_diag: dict, arch: str = "H3_cortex_agi") -> list[dict]:
    """Diagnose validation bottlenecks and emit falsifiable architectural/algorithmic candidates (including negative ablation controls)."""
    ax = base_diag["per_axis"]
    weakest = min(ax, key=ax.get)
    cands = [
        {
            "id": "ablation_no_surprise_gate",
            "hypothesis": "Negative ablation control: forcing surprise_gain=0 tests whether associative prediction-error gating actually contributes to episodic & cross-demo OOD binding.",
            "cfg_overrides": {},
            "freeze_surprise_zero": True,
            "lr": 7.5e-4,
            "ponder_eval": 4,
            "verifier_select": False,
        },
        {
            "id": f"mut1_finer_microchunk_mem48_target_{weakest}",
            "hypothesis": f"Targeting bottleneck ({weakest}={ax[weakest]:.3f}): finer episodic write granularity (micro_chunk=8 vs 16) + expanded associative rank (mem_dim=48 vs 32) reduces intra-episode key collision.",
            "cfg_overrides": {"micro_chunk": 8, "mem_dim": 48},
            "freeze_surprise_zero": False,
            "lr": 7.5e-4,
            "ponder_eval": 4,
            "verifier_select": False,
        },
        {
            "id": "mut2_deep_recursive_curriculum_verifier",
            "hypothesis": "Training with deeper latent recurrence (ponder_steps=6, micro_chunk=8, mem_dim=48) + Verifier-Guided Self-Correction (ponder_eval=8, verifier_select=True) improves compositional & causal OOD deduction.",
            "cfg_overrides": {"micro_chunk": 8, "mem_dim": 48, "ponder_steps": 6},
            "freeze_surprise_zero": False,
            "lr": 8.0e-4,
            "ponder_eval": 8,
            "verifier_select": True,
        },
    ]
    if arch == "H4_cortex_v2":
        cands.append(
            {
                "id": f"mut3_denser_slot_landmarks_target_{weakest}",
                "hypothesis": f"Targeting bottleneck ({weakest}={ax[weakest]:.3f}): doubling landmark capacity (n_slots=96) with finer write granularity (micro_chunk=8) raises coverage of rare OOD bindings.",
                "cfg_overrides": {"n_slots": 96, "micro_chunk": 8},
                "freeze_surprise_zero": False,
                "lr": 7.5e-4,
                "ponder_eval": 4,
                "verifier_select": False,
            }
        )
    return cands


def run_autonomous_loop(steps: int = 2400, batch_size: int = 36, seed: int = 0, device: str = "cuda", arch: str = "H3_cortex_agi") -> dict:
    suffix = "" if arch == "H3_cortex_agi" else f"_{arch}"
    archive_path = ROOT / f"cortex_runs/evolution_archive{suffix}.jsonl"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    def record(entry: dict):
        with open(archive_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    base_cfg = CortexConfig()
    model_cls = ARCH_REGISTRY[arch]
    print(f"[AutoLoop] Evaluating Gen-0 Baseline {arch}...", flush=True)
    ckpt_p = ROOT / f"cortex_runs/{arch}.pt"
    if ckpt_p.exists():
        ck = torch.load(ckpt_p, map_location=device)
        champion = model_cls(CortexConfig(**ck["cfg"])).to(device)
        champion.load_state_dict(ck["model"])
    else:
        champion, _ = train_multitask_model(arch, base_cfg, steps=steps, batch_size=batch_size, seed=seed, device=device)

    safe_ok, safe_msg = check_safety_invariants(champion, device=device)
    champ_val = validation_score(champion, ponder_eval=base_cfg.ponder_steps, verifier_select=False, device=device)
    champ_id = f"gen0_{arch}_base"
    champ_meta = {"ponder_eval": base_cfg.ponder_steps, "verifier_select": False}

    record({
        "gen": 0, "id": champ_id, "accepted": True, "safety": safe_msg,
        "val": champ_val, "params": sum(p.numel() for p in champion.parameters()),
    })
    print(f"[AutoLoop] Gen-0 {champ_id}: val_ood={champ_val['val_ood_mean']:.4f}, val_ece={champ_val['val_ece']:.4f} ({safe_msg})", flush=True)

    candidates = generate_hypotheses_from_diagnostics(champ_val, arch=arch)
    for gen_idx, cand in enumerate(candidates, start=1):
        t0 = time.time()
        print(f"\n[AutoLoop] Gen-{gen_idx} Testing {cand['id']}: {cand['hypothesis']}", flush=True)
        cfg_c = replace(base_cfg, **cand["cfg_overrides"])
        torch.manual_seed(seed)
        model_c = model_cls(cfg_c).to(device)
        if cand["freeze_surprise_zero"]:
            for b in model_c.mem_blocks:
                b.attn.surprise_gain.data.zero_()
                b.attn.surprise_gain.requires_grad_(False)

        # Train candidate under identical step & batch budget (fresh model_c if freeze_surprise_zero else train_multitask_model)
        if cand["freeze_surprise_zero"]:
            # Run same training loop on the surprise-zeroed model
            opt = torch.optim.AdamW([p for p in model_c.parameters() if p.requires_grad], lr=cand["lr"], betas=(0.9, 0.95), weight_decay=0.05)
            rng = __import__("random").Random(seed)
            from cortex.tasks import TRAIN_SAMPLERS, collate_episodes
            import torch.nn.functional as F, math
            autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))
            for st in range(steps):
                cur_lr = cand["lr"] * (st + 1) / 150 if st < 150 else cand["lr"] * (0.1 + 0.45 * (1 + math.cos(math.pi * (st - 150) / max(1, steps - 150))))
                for g in opt.param_groups:
                    g["lr"] = cur_lr
                per_t = max(1, batch_size // len(TRAIN_SAMPLERS))
                eps = [fn(rng) for _, fn in TRAIN_SAMPLERS for _ in range(per_t)]
                b_ep = collate_episodes(eps, device=device)
                with autocast:
                    out = model_c(b_ep["x"], b_ep["rows"], b_ep["cols"], b_ep["modality"])
                    w_s = [k + 1 for k in range(len(out["step_logits"]))]
                    loss = sum(w * episode_balanced_ce(lg, b_ep["y"]) for w, lg in zip(w_s, out["step_logits"])) / sum(w_s)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_c.parameters(), 1.0)
                opt.step()
        else:
            model_c, _ = train_multitask_model(arch, cfg_c, steps=steps, batch_size=batch_size, lr=cand["lr"], seed=seed, device=device)

        safe_ok, safe_msg = check_safety_invariants(model_c, device=device)
        val_c = validation_score(model_c, ponder_eval=cand["ponder_eval"], verifier_select=cand["verifier_select"], device=device)

        # Strict acceptance criterion: safety invariants pass, calibration ECE <= 0.20, AND val_ood_mean strictly beats champion by >= 0.005
        improved = safe_ok and (val_c["val_ece"] <= 0.20) and (val_c["val_ood_mean"] >= champ_val["val_ood_mean"] + 0.005)
        entry = {
            "gen": gen_idx,
            "id": cand["id"],
            "hypothesis": cand["hypothesis"],
            "params": sum(p.numel() for p in model_c.parameters()),
            "safety": safe_msg,
            "val": val_c,
            "delta_val_ood": round(val_c["val_ood_mean"] - champ_val["val_ood_mean"], 4),
            "accepted": improved,
            "elapsed_s": round(time.time() - t0, 1),
        }
        record(entry)
        status_str = "PROMOTED" if improved else "REJECTED (rolled back to champion)"
        print(
            f"[AutoLoop] Gen-{gen_idx} {cand['id']}: val_ood={val_c['val_ood_mean']:.4f} "
            f"(delta={entry['delta_val_ood']:+.4f}), ece={val_c['val_ece']:.4f} -> {status_str}",
            flush=True,
        )
        if improved:
            champion = model_c
            champ_val = val_c
            champ_id = cand["id"]
            champ_meta = {"ponder_eval": cand["ponder_eval"], "verifier_select": cand["verifier_select"], "cfg": asdict(cfg_c)}
            torch.save({"id": champ_id, "meta": champ_meta, "model": champion.state_dict()}, ROOT / f"cortex_runs/champion_evolved{suffix}.pt")

    # Final unbiased evaluation of the promoted champion on the UNTOUCHED Held-Out Test Split (seed=2026)
    print(f"\n[AutoLoop] Evaluating final champion ({champ_id}) on Held-Out Test Split...", flush=True)
    test_scores = evaluate_all_axes(champion, arch, device=device)
    ttt_scores = evaluate_arc_ttt(champion, n_episodes=80, device=device)
    final_summary = {
        "champion_id": champ_id,
        "champion_meta": champ_meta,
        "champion_val": champ_val,
        "heldout_test_benchmarks": test_scores,
        "heldout_arc_ood_ttt": ttt_scores,
    }
    (ROOT / f"cortex_runs/evolution_summary{suffix}.json").write_text(json.dumps(final_summary, indent=2))
    print(
        f"[AutoLoop] CHAMPION {champ_id}: Held-Out ID={test_scores['composite_id_acc']:.4f} | "
        f"Held-Out OOD={test_scores['composite_ood_acc']:.4f} | "
        f"ARC-OOD TTT Cell={ttt_scores['ttt_cell_acc']:.4f}",
        flush=True,
    )
    return final_summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--batch", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="H3_cortex_agi", choices=list(ARCH_REGISTRY.keys()))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_autonomous_loop(steps=args.steps, batch_size=args.batch, seed=args.seed, device=dev, arch=args.arch)
