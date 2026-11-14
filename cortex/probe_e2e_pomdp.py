"""Exp #036 — End-to-end learned POMDP: BeliefEncoder (learned tracker) feeds
the particle-belief EV beam (Exp #034). Replaces the rule-based `know` dict
with cells where the learned belief is confident (max softmax > tau) —
everything downstream (particle sampling, beam, EV) is identical.

Comparison anchors: rule-tracker + particle beam gave R2 72.5% / R3 87.5%.
"""
import argparse
import json
import random
from pathlib import Path

import torch

from cortex.belief_beam import sample_particles, select_action_belief
from cortex.belief_encoder import BeliefEncoder, fog_cells
from cortex.discrete_wm import DiscreteBoardWM, parse_state
from cortex.planning import KeyDoorTrapEnv

FOG = 8


def run_e2e_map(wm, be, env, R, K, tau, device, max_steps=18, rng_seed=0, soft=True):
    rng = random.Random(rng_seed)
    obs = env.reset()
    ret, n = 0.0, 0
    belief = None
    while not env.done and n < max_steps:
        cells_t, flag = parse_state(obs[0])
        pobs = fog_cells(cells_t, env.agent, R)
        po = torch.tensor(pobs, device=device).unsqueeze(0)
        if belief is None:
            belief = be.init_belief(po)
        else:
            act = torch.tensor([prev_a], device=device)
            logits, _ = be(belief, po, act)
            if soft:
                belief = torch.nn.functional.softmax(logits, -1)
            else:
                belief = torch.nn.functional.one_hot(logits.argmax(-1), 9).float()
        conf = belief.max(-1).values[0]
        ids = belief.argmax(-1)[0]
        know = {}
        for j in range(36):
            v = ids[j].item()
            if conf[j].item() > tau and v < FOG:
                know[divmod(j, 6)] = v
        # agent cell is always observed
        ar, ac = env.agent
        know[(ar, ac)] = 0
        particles = sample_particles(know, flag, env.agent, K, rng)
        know_p = [dict(know) for _ in particles]
        scores = select_action_belief(wm, particles, know_p, flag, R, device=device)
        a = max(scores, key=scores.get) if scores else rng.randrange(4)
        prev_a = a
        obs, r, done, trapped = env.step(a)
        ret += r
        n += 1
    return env.agent == env.goal and env.door_open, env.trapped, ret, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm-ckpt", default="cortex_runs/discrete_wm_v11.pt")
    ap.add_argument("--be-ckpt", default="cortex_runs/belief_enc.pt")
    ap.add_argument("--n-maps", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--radii", default="2")
    ap.add_argument("--particles", type=int, default=6)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--hard-feedback", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="cortex_runs/e2e_pomdp.json")
    a = ap.parse_args()

    wm = DiscreteBoardWM().to(a.device)
    wm.load_state_dict(torch.load(a.wm_ckpt, map_location=a.device))
    wm.eval()
    be = BeliefEncoder().to(a.device)
    be.load_state_dict(torch.load(a.be_ckpt, map_location=a.device))
    be.eval()

    report = {}
    for R in [int(x) for x in a.radii.split(",")]:
        g = t = 0
        rets, ns = [], []
        for m in range(a.n_maps):
            env = KeyDoorTrapEnv(a.seed0 + m)
            goal, trapped, ret, n = run_e2e_map(wm, be, env, R, a.particles, a.tau, a.device, rng_seed=1000 + m, soft=not a.hard_feedback)
            g += int(goal); t += int(trapped); rets.append(ret); ns.append(n)
        report[f"R{R}"] = {"goal_rate": round(g / a.n_maps, 4), "trap_rate": round(t / a.n_maps, 4),
                          "mean_return": round(sum(rets) / len(rets), 4),
                          "mean_steps": round(sum(ns) / len(ns), 2)}
        print(f"[e2e R={R}] {report[f'R{R}']}", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
