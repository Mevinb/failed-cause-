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

from cortex.belief_beam import knowledge_from_obs, sample_particles, select_action_belief
from cortex.belief_encoder import BeliefEncoder, fog_cells
from cortex.discrete_wm import DiscreteBoardWM, parse_state
from cortex.planning import KeyDoorTrapEnv

FOG = 8


def run_e2e_map(wm, be, env, R, K, tau, device, max_steps=18, rng_seed=0, soft=True, stale_T: int = 5, uniform: bool = False):
    rng = random.Random(rng_seed)
    obs = env.reset()
    ret, n = 0.0, 0
    belief = None
    know: dict = {}
    last_seen: dict = {}
    while not env.done and n < max_steps:
        cells_t, flag = parse_state(obs[0])
        pobs = fog_cells(cells_t, env.agent, R)
        knowledge_from_obs(know, pobs, env.agent, R)
        ar, ac = env.agent
        for r in range(6):
            for c in range(6):
                if abs(r - ar) + abs(c - ac) <= R:
                    last_seen[(r, c)] = n
        # consumed landmarks are gone from the world: purge stale knowledge
        if flag & 1:
            know = {k: v for k, v in know.items() if v != 3}
        if flag & 2:
            know = {k: v for k, v in know.items() if v != 4}
        # moving world: landmark knowledge decays — a landmark not re-observed
        # within stale_T steps may have relocated unseen; statics persist.
        know = {
            k: v for k, v in know.items()
            if v < 3 or n - last_seen.get(k, -10 ** 9) <= stale_T
        }
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
        prior = None if uniform else belief[0].tolist()  # [36,9] learned per-cell type prior for unseen cells
        particles = sample_particles(know, flag, env.agent, K, rng, prior=prior)
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
    ap.add_argument("--move-prob", type=float, default=0.0)
    ap.add_argument("--trap-hunt", type=float, default=0.0)
    ap.add_argument("--stale-T", type=int, default=5)
    ap.add_argument("--uniform-prior", action="store_true")
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
            env = KeyDoorTrapEnv(a.seed0 + m, move_prob=a.move_prob, trap_hunt=a.trap_hunt)
            goal, trapped, ret, n = run_e2e_map(wm, be, env, R, a.particles, a.tau, a.device, rng_seed=1000 + m, soft=not a.hard_feedback, stale_T=a.stale_T, uniform=a.uniform_prior)
            g += int(goal); t += int(trapped); rets.append(ret); ns.append(n)
            print(f"[R={R} map {m}] goal={goal} trap={trapped} ret={ret:.2f} steps={n}", flush=True)
        report[f"R{R}"] = {"goal_rate": round(g / a.n_maps, 4), "trap_rate": round(t / a.n_maps, 4),
                          "mean_return": round(sum(rets) / len(rets), 4),
                          "mean_steps": round(sum(ns) / len(ns), 2)}
        print(f"[e2e R={R}] {report[f'R{R}']}", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
