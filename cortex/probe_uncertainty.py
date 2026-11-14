"""Exp #032 — Planning under state uncertainty (Wall-1 x Wall-2 bridge).
The v11 planner hits the oracle ceiling when it can SEE the bait trap. This
probe hides the trap cell from its observations (simulating a landmark never
yet observed) and asks: does maximin belief-planning rescue it?

  visible  : normal obs (trap shown)          — the 90% baseline
  masked   : trap cell -> empty in every obs  — planner ignores the trap
  maximin  : same masked obs, but candidates = free cells adjacent to the start
             (env always places the trap adjacent to the agent at t0); each
             step builds one belief board per surviving candidate and takes
             argmax_a min_belief score(a). Survival on a cell refutes it.

Metric: goal_rate / trap_rate / mean_return per mode on held-out maps.
"""
import argparse
import json
from pathlib import Path

import torch

from cortex.discrete_wm import DiscreteBoardWM, decode_state, onehot_from_parsed, parse_state, select_action_learned
from cortex.planning import ACTIONS, COLOR_BASE, KeyDoorTrapEnv


def masked_obs(obs, trap_rc):
    toks = list(obs[0])
    toks[3 + trap_rc[0] * 6 + trap_rc[1]] = COLOR_BASE + 0
    return (toks, obs[1], obs[2])


def belief_s0(wm, obs_toks, cand, device):
    cells, flag = parse_state(obs_toks)
    cells[cand] = 6
    return onehot_from_parsed(torch.tensor([cells], dtype=torch.long, device=device), torch.tensor([flag], dtype=torch.long, device=device))


def run_map(wm, env, mode, device, max_steps=18):
    obs = env.reset()
    cells0, _ = parse_state(obs[0])
    ag0 = cells0.index(2)
    r0, c0 = divmod(ag0, 6)
    cands = [
        r * 6 + c
        for dr, dc in ACTIONS[:4]
        if 0 <= (r := r0 + dr) < 6 and 0 <= (c := c0 + dc) < 6 and cells0[r * 6 + c] != 1
    ]
    ret, n = 0.0, 0
    while not env.done and n < max_steps:
        if mode == "visible":
            a = select_action_learned(wm, obs[0], device=device, mode="raw")
        else:
            mobs = masked_obs(obs, env.trap)
            if mode == "masked":
                a = select_action_learned(wm, mobs[0], device=device, mode="raw")
            else:  # maximin
                if not cands:
                    a = select_action_learned(wm, mobs[0], device=device, mode="raw")
                else:
                    per_a = {a_: 1e9 for a_ in range(4)}
                    for c in cands:
                        s0 = belief_s0(wm, mobs[0], c, device)
                        sc = select_action_learned(wm, mobs[0], device=device, mode="raw", s0=s0, return_scores=True)
                        for a_ in range(4):
                            per_a[a_] = min(per_a[a_], sc.get(a_, -1e9))
                    a = max(per_a, key=per_a.get)
        obs, r, done, trapped = env.step(a)
        ret += r
        n += 1
        if mode == "maximin" and not trapped:
            ar, ac = env.agent
            cands = [c for c in cands if c != ar * 6 + ac]
    return env.agent == env.goal and env.door_open, env.trapped, ret, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="cortex_runs/discrete_wm_v11.pt")
    ap.add_argument("--n-maps", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="cortex_runs/uncertainty_probe.json")
    a = ap.parse_args()

    wm = DiscreteBoardWM().to(a.device)
    wm.load_state_dict(torch.load(a.ckpt, map_location=a.device))
    wm.eval()

    report = {}
    for mode in ["visible", "masked", "maximin"]:
        g = t = 0
        rets = []
        for m in range(a.n_maps):
            env = KeyDoorTrapEnv(a.seed0 + m)
            goal, trapped, ret, n = run_map(wm, env, mode, a.device)
            g += int(goal); t += int(trapped); rets.append(ret)
        report[mode] = {"goal_rate": round(g / a.n_maps, 4), "trap_rate": round(t / a.n_maps, 4),
                        "mean_return": round(sum(rets) / len(rets), 4)}
        print(f"[{mode}] {report[mode]}", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
