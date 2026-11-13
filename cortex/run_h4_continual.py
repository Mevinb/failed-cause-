"""Standalone Gen-2 continual eval: runs the 4-regime 5-phase experiment on H4_cortex_v2
(the in-flight suite used the pre-edit snapshot which only evaluates H3)."""
import json, sys
sys.path.insert(0, "/home/mevlec/Data/exp/agi-exi")
from cortex.continual import run_continual_experiment
from cortex.arch import H4CortexV2

res = run_continual_experiment(steps_per_phase=420, batch_size=32, device="cuda", model_cls=H4CortexV2)
for reg, vals in res.items():
    print(f"[Continual:H4_cortex_v2:{reg:<18}] FAA={vals['final_avg_acc']:.4f} | BWT={vals['backward_transfer']:+.4f} | "
          f"P5_Forward={vals['phase5_forward_acc']:.4f} | Dormant={vals['dormant_unit_frac']:.4f}", flush=True)
json.dump(res, open("cortex_runs/continual_H4_gen2.json", "w"), indent=2)
print("WROTE cortex_runs/continual_H4_gen2.json")
