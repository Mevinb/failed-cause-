"""Isolation ablation: does sinusoidal depth encoding ALONE drive the causal grok?
Trains H3_sinusoidal_abl (H3 memory + sinusoidal reasoner, NO slots) seed 0 & 1."""
import json, sys, torch
sys.path.insert(0, "/home/mevlec/Data/exp/agi-exi")
from cortex.arch import CortexConfig
from cortex.run_suite import train_multitask_model, eval_split
from cortex.tasks import sample_causal, sample_episodic, sample_s5

dev = "cuda"
out = {}
for seed in [0, 1]:
    model, _ = train_multitask_model("H3_sinusoidal_abl", CortexConfig(), steps=3600, batch_size=36, seed=seed, device=dev)
    torch.save({"arch": "H3_sinusoidal_abl", "seed": seed, "model": model.state_dict()}, f"cortex_runs/H3_sinusoidal_abl_seed{seed}.pt")
    ax = [("causal_id", sample_causal, "id"), ("causal_ood_do", sample_causal, "ood_do_only"),
          ("causal_ood_scm", sample_causal, "ood_scm"), ("episodic_ood_gap160", sample_episodic, "long160"),
          ("s5_ood_len10", sample_s5, "len10")]
    out[f"seed{seed}"] = {n: eval_split(model, f, s, n_eval=240, seed=2026, device=dev)["exact_acc"] for n, f, s in ax}
    print(f"H3sin_seed{seed}:", json.dumps(out[f"seed{seed}"]), flush=True)
json.dump(out, open("cortex_runs/h3_sinusoidal_ablation.json", "w"), indent=2)
