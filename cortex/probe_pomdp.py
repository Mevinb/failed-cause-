"""Exp #033 — POMDP probe: memory-fed imagination under fog-of-war.
The env stays fully dynamic; the agent only observes cells within Manhattan
radius R (fog). A rule-based belief tracker (the "memory" subsystem) accumulates
every observed cell — landmarks/walls/trap are static so first-seen is permanent;
flag flips handle consumption. The belief board is rendered back into obs format
(unseen cells = optimistic empty) and fed to the SAME learned beam planner.

Policy:
  - if the current milestone target is present in the belief board -> learned beam
  - else -> frontier exploration: BFS on the belief board to the nearest
    unexplored cell (information-seeking as a hand-rolled primitive — the
    scientific question is whether memory+imagination+frontier suffices).

Metrics vs R: goal_rate / trap_rate / mean_return / mean_steps.
"""
import argparse
import json
from collections import deque
from pathlib import Path

import torch

from cortex.discrete_wm import DiscreteBoardWM, parse_state, select_action_learned
from cortex.planning import ACTIONS, COLOR_BASE, KeyDoorTrapEnv
from cortex.tasks import TOK_IN_GRID, TOK_RESET


def belief_obs(seen: dict, flag: int, agent_rc: tuple, size: int = 6):
    """Render accumulated memory into the obs token layout (unseen cells -> 0)."""
    toks = [TOK_RESET, TOK_IN_GRID, COLOR_BASE + flag]
    for r in range(size):
        for c in range(size):
            if (r, c) == agent_rc:
                toks.append(COLOR_BASE + 2)
            else:
                toks.append(COLOR_BASE + seen.get((r, c), 0))
    return toks


def update_seen(seen: dict, obs_cells: list[int], agent_rc: tuple, R: int):
    """Add every cell within Manhattan-R of the agent to memory."""
    ar, ac = agent_rc
    for r in range(6):
        for c in range(6):
            if abs(r - ar) + abs(c - ac) <= R:
                v = obs_cells[r * 6 + c]
                if v != 2:  # agent overlay: the underlying cell is what's remembered
                    seen[(r, c)] = v
                elif (r, c) not in seen:
                    seen[(r, c)] = 0


def belief_cells(seen: dict, flag: int) -> list[int]:
    """Board cells from memory with flag-consumption applied (key/door vanish)."""
    g = [0] * 36
    for (r, c), v in seen.items():
        g[r * 6 + c] = v
    if flag & 1:
        g = [0 if v == 3 else v for v in g]
    if flag & 2:
        g = [0 if v == 4 else v for v in g]
    return g


def frontier_action(seen: dict, cells: list[int], agent_rc: tuple):
    """BFS on the belief board to the nearest unexplored cell; return first action."""
    ar, ac = agent_rc
    q = deque([(agent_rc, None)])
    visited = {agent_rc}
    target = None
    while q:
        (r, c), first = q.popleft()
        if (r, c) not in seen:
            target = (r, c, first)
            break
        for a_i, (dr, dc) in enumerate(ACTIONS[:4]):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 6 and 0 <= nc < 6 and (nr, nc) not in visited and cells[nr * 6 + nc] not in (1, 6):
                visited.add((nr, nc))
                q.append(((nr, nc), a_i if first is None else first))
    return target[2] if target else 0


def run_pomdp_map(wm, env, R, device, max_steps=18):
    obs = env.reset()
    seen: dict = {}
    ret, n = 0.0, 0
    while not env.done and n < max_steps:
        cells_true, flag = parse_state(obs[0])
        ar, ac = env.agent
        update_seen(seen, cells_true, (ar, ac), R)
        bcells = belief_cells(seen, flag)
        # milestone target visible in belief?
        hk, do = flag & 1, flag & 2
        tgt_cell = 3 if not hk else (4 if not do else 5)
        if tgt_cell in bcells:
            toks = belief_obs(seen, flag, (ar, ac))
            a = select_action_learned(wm, toks, device=device, mode="raw")
        else:
            a = frontier_action(seen, bcells, (ar, ac))
        obs, r, done, trapped = env.step(a)
        ret += r
        n += 1
    return env.agent == env.goal and env.door_open, env.trapped, ret, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="cortex_runs/discrete_wm_v11.pt")
    ap.add_argument("--n-maps", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--radii", default="2,3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="cortex_runs/pomdp_probe.json")
    a = ap.parse_args()

    wm = DiscreteBoardWM().to(a.device)
    wm.load_state_dict(torch.load(a.ckpt, map_location=a.device))
    wm.eval()

    report = {}
    for R in [int(x) for x in a.radii.split(",")]:
        g = t = 0
        rets = []
        ns = []
        for m in range(a.n_maps):
            env = KeyDoorTrapEnv(a.seed0 + m)
            goal, trapped, ret, n = run_pomdp_map(wm, env, R, a.device)
            g += int(goal); t += int(trapped); rets.append(ret); ns.append(n)
        report[f"R{R}"] = {"goal_rate": round(g / a.n_maps, 4), "trap_rate": round(t / a.n_maps, 4),
                          "mean_return": round(sum(rets) / len(rets), 4),
                          "mean_steps": round(sum(ns) / len(ns), 2)}
        print(f"[R={R}] {report[f'R{R}']}", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
