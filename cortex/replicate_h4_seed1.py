"""Seed-1 replication of the H4 causal/episodic gains (Exp #015 follow-up).
Rules out a lucky init lottery behind the 0.98 do-calculus jump."""
import json, sys, torch
sys.path.insert(0, "/home/mevlec/Data/exp/agi-exi")
from cortex.arch import CortexConfig
from cortex.run_suite import train_multitask_model, eval_split
from cortex.tasks import sample_causal, sample_episodic, sample_multimodal, sample_s5

dev = "cuda"
model, hist = train_multitask_model("H4_cortex_v2", CortexConfig(), steps=3600, batch_size=36, seed=1, device=dev)
torch.save({"arch": "H4_cortex_v2", "seed": 1, "model": model.state_dict()}, "cortex_runs/H4_cortex_v2_seed1.pt")

AXES = [("causal_id_mixed", sample_causal, "id"), ("causal_ood_do", sample_causal, "ood_do_only"),
        ("causal_ood_scm", sample_causal, "ood_scm"), ("episodic_ood_gap160", sample_episodic, "long160"),
        ("mm_ood_2hop", sample_multimodal, "hops2"), ("s5_ood_len10", sample_s5, "len10")]
res = {name: eval_split(model, fn, sp, n_eval=240, seed=2026, device=dev)["exact_acc"] for name, fn, sp in AXES}
print("H4_seed1:", json.dumps(res))
json.dump(res, open("cortex_runs/h4_seed1_replication.json", "w"), indent=2)
