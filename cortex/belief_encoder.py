"""Exp #035 — Learned belief encoder: history-conditioned state estimation.

Exp #033's POMDP result used a RULE-BASED belief tracker (accumulate seen cells).
This is the learned version: a recurrent conv net that maintains a belief board
B_t (36 x 9 types: 8 cell types + UNSEEN) updated from (B_{t-1}, partial obs_t,
action_t). Supervision: the true board at each step of fog-R trajectories.

The learned tracker must discover the same structure (statics persist, flag
flips consume, agent moves) from data — and unlike the rule tracker it can
represent UNCERTAINTY (softmax over unseen cells) rather than committing to an
optimistic fill. Beam planning then consumes the belief's argmax (or, later,
samples over the belief — particle-planner without hand sampling).

Train+eval: .venv/bin/python -m cortex.belief_encoder --steps 6000 --device cuda
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex.discrete_wm import parse_state
from cortex.planning import ACTIONS, COLOR_BASE, KeyDoorTrapEnv

GRID, N_CELL, N_FLAG, N_ACT = 36, 8, 4, 4
FOG = N_CELL  # type id 8 = unseen
N_BEL = N_CELL + 1


def fog_cells(true_cells: list[int], agent_rc: tuple, R: int) -> list[int]:
    """partial observation: cells beyond Manhattan-R read as FOG (8)."""
    ar, ac = agent_rc
    out = []
    for i, v in enumerate(true_cells):
        r, c = divmod(i, 6)
        out.append(v if abs(r - ar) + abs(c - ac) <= R else FOG)
    return out


def collect_belief_dataset(n_envs: int, R: int, seed_start: int = 20000, move_prob: float = 0.0) -> list[dict]:
    """Fog-R trajectories with (partial_obs, action, true_next_belief) records."""
    rng = random.Random(seed_start)
    eps = []
    for i in range(n_envs):
        env = KeyDoorTrapEnv(seed_start + i, move_prob=move_prob)
        obs = env.reset()
        seq = []
        for t in range(len(env.optimal_actions)):
            cells_t, flag_t = parse_state(obs[0])
            ar, ac = env.agent
            a = env.optimal_actions[t]
            seq.append({
                "pobs": fog_cells(cells_t, (ar, ac), R),
                "act": a,
                "flag": flag_t,
            })
            obs, r, done, trapped = env.step(a)
            cells_n, flag_n = parse_state(obs[0])
            seq[-1]["true_next"] = cells_n
            seq[-1]["flag_next"] = flag_n
            if done:
                break
        eps.append(seq)
    return eps


class BeliefEncoder(nn.Module):
    """Belief board recurrence: B_t = f(B_{t-1}, fog_obs_t, act_t).
    B is [B, 36, 9] one-hot/prob over cell types + unseen. Conv trunk on the
    6x6 grid (equivariant, like the world model). Output = per-cell logits
    over 9 belief types + flag head."""

    def __init__(self, d: int = 128):
        super().__init__()
        cin = N_BEL + N_BEL + N_ACT
        self.trunk = nn.Sequential(
            nn.Conv2d(cin, d, 3, padding=1), nn.SiLU(),
            nn.Conv2d(d, d, 3, padding=1), nn.SiLU(),
            nn.Conv2d(d, d, 3, padding=1), nn.SiLU(),
        )
        self.head = nn.Conv2d(d, N_BEL, 1)
        self.head_flag = nn.Sequential(nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, N_FLAG))

    def init_belief(self, pobs: torch.Tensor) -> torch.Tensor:
        """first obs (partial) -> initial belief: observed cells certain,
        unobserved = FOG."""
        return F.one_hot(pobs, N_BEL).float()

    def forward(self, belief: torch.Tensor, pobs: torch.Tensor, act: torch.Tensor) -> tuple:
        """belief [B,36,9] probs, pobs [B,36] ids, act [B] -> (belief_logits, flag_logits)."""
        B = belief.shape[0]
        bg = belief.transpose(1, 2).reshape(B, N_BEL, 6, 6)
        og = F.one_hot(pobs, N_BEL).float().transpose(1, 2).reshape(B, N_BEL, 6, 6)
        ag = F.one_hot(act, N_ACT).float().view(B, N_ACT, 1, 1).expand(B, N_ACT, 6, 6)
        h = self.trunk(torch.cat([bg, og, ag], dim=1))
        logits = self.head(h).flatten(2).transpose(1, 2)
        flag = self.head_flag(h.mean(dim=(2, 3)))
        return logits, flag


def belief_to_board(belief_logits: torch.Tensor) -> list[int]:
    """argmax belief -> board cells; FOG cells -> 0 (optimistic)."""
    ids = belief_logits.argmax(-1)[0].tolist()
    return [v if v < N_CELL else 0 for v in ids]


def train_belief(be: BeliefEncoder, eps: list[dict], steps: int, bs: int, R: int, device: str, seed: int = 0):
    rng = random.Random(seed)
    be.to(device).train()
    opt = torch.optim.AdamW(be.parameters(), lr=6e-4)
    for it in range(steps):
        # sample (episode, t) pairs; unroll 2 steps for recurrence consistency
        batch = []
        while len(batch) < bs:
            ep = eps[rng.randrange(len(eps))]
            if len(ep) >= 2:
                batch.append((ep, rng.randrange(len(ep) - 1)))
        loss = 0.0
        acc_n = acc_t = 0
        for ep, t in batch:
            pobs0 = torch.tensor(ep[t]["pobs"], device=device).unsqueeze(0)
            b = be.init_belief(pobs0)
            seen = torch.zeros(GRID, dtype=torch.bool, device=device)
            for kk in range(t):
                seen |= torch.tensor(ep[kk]["pobs"], device=device) != FOG
            for u in range(2):
                k = t + u
                po = torch.tensor(ep[k]["pobs"], device=device).unsqueeze(0)
                seen |= po[0] != FOG
                ac = torch.tensor([ep[k]["act"]], device=device)
                logits, flag_l = be(b, po, ac)
                true_next = torch.tensor(ep[k]["true_next"], device=device)
                tgt = torch.where(seen, true_next, torch.full_like(true_next, FOG)).unsqueeze(0)
                loss = loss + F.cross_entropy(logits.reshape(-1, N_BEL), tgt.reshape(-1))
                loss = loss + 0.3 * F.cross_entropy(flag_l, torch.tensor([ep[k]["flag_next"]], device=device))
                acc_n += int((logits.argmax(-1) == tgt).sum().item())
                acc_t += GRID
                b = F.one_hot(logits.argmax(-1), N_BEL).float().detach()  # scheduled-sampling style
        loss = loss / bs
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 250 == 0:
            print(f"[belief] step {it+1} loss {loss.item():.4f} cell_acc {acc_n/acc_t:.4f}", flush=True)
    be.eval()
    return be


def eval_belief_accuracy(be: BeliefEncoder, eps: list[dict], R: int, device: str, n: int = 200) -> dict:
    """Roll the belief forward through whole episodes; measure board accuracy on
    KNOWN cells and how well uncertainty marks truly-unknown content."""
    tot = hit = 0
    seen_hit = seen_tot = fog_mark_seen = fog_mark_unseen = fog_n = 0
    rng = random.Random(123)
    for i in range(min(n, len(eps))):
        ep = eps[i]
        b = be.init_belief(torch.tensor(ep[0]["pobs"], device=device).unsqueeze(0))
        seen = torch.zeros(GRID, dtype=torch.bool, device=device)
        for k, rec in enumerate(ep):
            po = torch.tensor(rec["pobs"], device=device).unsqueeze(0)
            seen |= po[0] != FOG
            ac = torch.tensor([rec["act"]], device=device)
            logits, fl = be(b, po, ac)
            pred = logits.argmax(-1)[0]
            true = rec["true_next"]
            for j in range(GRID):
                tot += 1
                hit += int(pred[j].item() == true[j] or (pred[j].item() == FOG and not seen[j]))
                if seen[j]:
                    seen_tot += 1
                    seen_hit += int(pred[j].item() == true[j])
                elif pred[j].item() == FOG:
                    fog_mark_unseen += 1
                else:
                    fog_mark_seen += 1  # committed a guess on an unseen cell
            fog_n += int((pred == FOG).sum().item())
            b = F.one_hot(logits.argmax(-1), N_BEL).float().detach()
    return {"board_acc": hit / max(1, tot), "seen_acc": seen_hit / max(1, seen_tot),
            "unseen_fog_recall": fog_mark_unseen / max(1, fog_mark_unseen + fog_mark_seen),
            "mean_fog_cells": fog_n / max(1, len(eps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--n-envs", type=int, default=3000)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--move-prob", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-out", default="cortex_runs/belief_enc.pt")
    ap.add_argument("--out", default="cortex_runs/belief_enc_eval.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    eps = collect_belief_dataset(a.n_envs, a.R, move_prob=a.move_prob)
    held = collect_belief_dataset(300, a.R, seed_start=60000, move_prob=a.move_prob)
    print(f"[belief] {sum(len(e) for e in eps)} steps of fog-R={a.R} data", flush=True)
    be = train_belief(BeliefEncoder(), eps, a.steps, a.bs, a.R, a.device, a.seed)
    torch.save(be.state_dict(), a.ckpt_out)
    res = eval_belief_accuracy(be, held, a.R, a.device)
    print(json.dumps(res, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
