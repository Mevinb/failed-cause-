"""Exp #034 — Particle-belief EV beam: information-seeking emerges from lookahead.

State under fog-of-war is a BELIEF: a set of known cells (knowledge) + priors
over the rest. We approximate it with K sampled board-particles consistent
with what the agent has actually observed. Inside the beam's imagined rollout,
each particle evolves under the LEARNED transition T; when the imagined agent
would observe a cell (Manhattan distance <= R), the agent's per-particle
knowledge updates from that particle's ground truth. Scoring a node therefore
rewards imagined moves that REVEAL the milestone target: in particles where
the imagined reveal found the key/door/goal, subsequent steps get BFS shaping
toward it -> higher node score -> EV over particles prefers informative moves.
This is genuine POMDP lookahead (value-of-information) rather than the
hand-rolled frontier of Exp #033.

At each real step: real obs reveals cells -> knowledge update -> particles
resampled/filtered for consistency -> batched particle beam -> EV pick.
"""
import random
from collections import deque

import torch

from cortex.discrete_wm import DiscreteBoardWM, decode_state, onehot_from_parsed, parse_state
from cortex.planning import ACTIONS, COLOR_BASE, KeyDoorTrapEnv

GAMMA = 0.97


def bfs_dist(cells: list[int], start: int, blocked_types=(1,)) -> list[int]:
    d = [-1] * 36
    d[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        r, c = divmod(u, 6)
        for dr, dc in ACTIONS[:4]:
            nr, nc = r + dr, c + dc
            v = nr * 6 + nc
            if 0 <= nr < 6 and 0 <= nc < 6 and d[v] < 0 and cells[v] not in blocked_types:
                d[v] = d[u] + 1
                q.append(v)
    return d


def knowledge_from_obs(know: dict, obs_cells: list[int], agent_rc: tuple, R: int):
    """Real-step update: cells within R of the agent enter knowledge. The cell
    under the agent reads as '2' — record it as 0 (statics handle landmarks)."""
    ar, ac = agent_rc
    for r in range(6):
        for c in range(6):
            if abs(r - ar) + abs(c - ac) <= R:
                v = obs_cells[r * 6 + c]
                know[(r, c)] = 0 if v == 2 else v


def milestone_target(know_cells: dict, flag: int) -> int:
    """cell index of the current milestone target if its position is KNOWN, else -1."""
    hk, do = flag & 1, flag & 2
    want = 3 if not hk else (4 if not do else 5)
    for (r, c), v in know_cells.items():
        if v == want:
            return r * 6 + c
    return -1


def sample_particles(know: dict, flag: int, agent_rc: tuple, K: int, rng: random.Random, n_walls: int = 5):
    """K board samples consistent with knowledge: seen cells fixed; unseen
    landmark cells (key/door/goal not yet consumed, unseen walls, unseen trap)
    drawn uniformly over unseen non-wall cells."""
    parts = []
    for _ in range(K):
        g = [0] * 36
        seen_walls = sum(1 for v in know.values() if v == 1)
        for (r, c), v in know.items():
            g[r * 6 + c] = v
        free = [i for i in range(36) if (i // 6, i % 6) not in know]
        # fill remaining walls + missing landmarks from the prior
        need = [1] * max(0, n_walls - seen_walls)
        present = set(know.values())
        if not (flag & 1) and 3 not in present:
            need.append(3)
        if not (flag & 2) and 4 not in present:
            need.append(4)
        if 5 not in present:
            need.append(5)
        if 6 not in present:
            need.append(6)
        for v in need:
            if not free:
                break
            j = rng.choice(free)
            g[j] = v
            free.remove(j)
        ag = agent_rc[0] * 6 + agent_rc[1]
        g[ag] = 2
        parts.append(g)
    return parts


def reveal_update(know_p: list[dict], boards: list[list[int]], R: int):
    """Per-particle imagined observation: for each particle, cells within R of
    the imagined agent enter that particle's knowledge."""
    for p, g in enumerate(boards):
        if 2 not in g:
            continue
        ar, ac = divmod(g.index(2), 6)
        for r in range(6):
            for c in range(6):
                if abs(r - ar) + abs(c - ac) <= R:
                    know_p[p][(r, c)] = g[r * 6 + c]


@torch.no_grad()
def select_action_belief(wm: DiscreteBoardWM, particles_cells: list[list[int]], know: list[dict],
                         flag: int, R: int, horizon: int = 14, beam_width: int = 48,
                         device: str = "cpu", voi_w: float = 0.15):
    """Batched-particle EV beam. Node = (K boards as tensor, K knowledge dicts,
    score, first_a, term). EV = mean over particles. voi_w weights the imagined
    reveal count (cells newly known per particle)."""
    K = len(particles_cells)
    flags_t = torch.full((K,), flag, dtype=torch.long, device=device)
    s0 = onehot_from_parsed(torch.tensor(particles_cells, dtype=torch.long, device=device), flags_t)
    k0 = [dict(kp) for kp in know]
    nodes = [(s0, k0, [0.0] * K, -1, False)]
    fa_scores: dict[int, float] = {}
    for h in range(horizon):
        cands = []
        for s, kp, sc_l, first_a, term in nodes:
            if term:
                continue
            for a in range(4):
                sn, r, tr = wm.step(s, torch.full((K,), a, device=device, dtype=torch.long), hard=True)
                cells_b = [decode_state(sn[i : i + 1])[0] for i in range(K)]
                know_new = [dict(k_) for k_ in kp]
                before = [len(k_) for k_ in know_new]
                reveal_update(know_new, cells_b, R)
                voi = [(len(know_new[i]) - before[i]) for i in range(K)]
                flag_n = decode_state(sn[0:1])[1]
                nsc = []
                term_any = False
                for i in range(K):
                    cells, fl = decode_state(sn[i : i + 1])
                    p_trap = torch.sigmoid(tr[i]).item()
                    trapped = 6 not in cells
                    solved = (fl >> 1) & 1 == 1 and 5 not in cells
                    sc = sc_l[i] + (GAMMA**h) * (r[i].item() - 4.0 * p_trap) + voi_w * voi[i]
                    if solved:
                        sc += 10.0
                    elif trapped:
                        sc -= 5.0
                    else:
                        ag = cells.index(2) if 2 in cells else -1
                        tgt = milestone_target(know_new[i], fl)
                        if ag >= 0 and tgt >= 0:
                            d = bfs_dist(cells, ag)
                            sc -= 0.06 * (d[tgt] if d[tgt] >= 0 else 12)
                        elif ag >= 0:
                            d = bfs_dist(cells, ag)
                            sc -= 0.06 * max(d) if max(d) > 0 else 0
                    nsc.append(sc)
                    term_any = term_any or trapped or solved
                fa = a if h == 0 else first_a
                cands.append((sum(nsc) / K, sn, know_new, nsc, fa, term_any, tuple(tuple(g) for g in cells_b)))
        cands.sort(key=lambda t: -t[0])
        seen, new_nodes = set(), []
        for msc, sn, kn, nsc, fa, term, dedup_k in cands:
            if dedup_k in seen:
                continue
            seen.add(dedup_k)
            new_nodes.append((sn, kn, nsc, fa, term))
            if fa >= 0:
                fa_scores[fa] = max(fa_scores.get(fa, -1e9), msc)
            if len(new_nodes) >= beam_width:
                break
        nodes = new_nodes
        if not nodes:
            break
    return fa_scores


def run_belief_map(wm, env, R, K, device, max_steps=18, rng_seed=0):
    rng = random.Random(rng_seed)
    obs = env.reset()
    know: dict = {}
    ret, n = 0.0, 0
    while not env.done and n < max_steps:
        cells_t, flag = parse_state(obs[0])
        knowledge_from_obs(know, cells_t, env.agent, R)
        # consumed landmarks are gone from the world: purge stale knowledge
        if flag & 1:
            know = {k: v for k, v in know.items() if v != 3}
        if flag & 2:
            know = {k: v for k, v in know.items() if v != 4}
        particles = sample_particles(know, flag, env.agent, K, rng)
        know_p = [dict(know) for _ in particles]
        scores = select_action_belief(wm, particles, know_p, flag, R, device=device)
        a = max(scores, key=scores.get) if scores else rng.randrange(4)
        obs, r, done, trapped = env.step(a)
        ret += r
        n += 1
    return env.agent == env.goal and env.door_open, env.trapped, ret, n


def main():
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="cortex_runs/discrete_wm_v11.pt")
    ap.add_argument("--n-maps", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--radii", default="2,3")
    ap.add_argument("--particles", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="cortex_runs/belief_beam.json")
    a = ap.parse_args()

    wm = DiscreteBoardWM().to(a.device)
    wm.load_state_dict(torch.load(a.ckpt, map_location=a.device))
    wm.eval()

    report = {}
    for R in [int(x) for x in a.radii.split(",")]:
        g = t = 0
        rets, ns = [], []
        for m in range(a.n_maps):
            env = KeyDoorTrapEnv(a.seed0 + m)
            goal, trapped, ret, n = run_belief_map(wm, env, R, a.particles, a.device, rng_seed=1000 + m)
            g += int(goal); t += int(trapped); rets.append(ret); ns.append(n)
            print(f"  [R={R}] map {m}: goal={goal} trapped={trapped} steps={n}", flush=True)
        report[f"R{R}_K{a.particles}"] = {"goal_rate": round(g / a.n_maps, 4), "trap_rate": round(t / a.n_maps, 4),
                                        "mean_return": round(sum(rets) / len(rets), 4),
                                        "mean_steps": round(sum(ns) / len(ns), 2)}
        print(f"[R={R},K={a.particles}] {report[f'R{R}_K{a.particles}']}", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
