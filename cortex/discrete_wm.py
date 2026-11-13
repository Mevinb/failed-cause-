"""Discrete-state world model — the Exp-#021-motivated test of whether a *discrete*
latent can hold faithful imagined rollouts where the continuous `z + trans(z,a)`
representation provably could not (72.6% agent fidelity at 1 step, 0% at 12).

State: the board itself — 36 cells x 8 types + milestone flag (4 values), fully
supervised from parsed observation tokens (exact ground truth, no decode ambiguity).
Transition: (one-hot state, action) -> next-state logits; hard argmax at rollout
time keeps the state on the discrete manifold BY CONSTRUCTION.

Run: .venv/bin/python cortex/discrete_wm.py [--steps 3000] [--eval-depth 14]
"""

import argparse
import json
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex.planning import ACTIONS, COLOR_BASE, KeyDoorTrapEnv, _bfs, collect_world_model_dataset

GRID = 36
N_CELL = 8
N_FLAG = 4
N_ACT = 5
STATE_DIM = GRID * N_CELL + N_FLAG


def parse_state(obs_toks: list[int]) -> tuple[list[int], int]:
    """obs -> discrete state: 36 cell ids + flag (has_key + 2*door_open)."""
    cells = [t - COLOR_BASE for t in obs_toks[3 : 3 + GRID]]
    flag = obs_toks[2] - COLOR_BASE
    return cells, flag


def state_to_tensor(cells: list[int], flag: int, device="cpu") -> torch.Tensor:
    return torch.tensor(cells + [N_CELL + flag], dtype=torch.long, device=device)


class DiscreteBoardWM(nn.Module):
    """Learned discrete world model. The encoder is a per-position shared head (the
    obs->board map is positional, so a shared head generalizes exactly on held-out
    boards where a flat MLP memorizes); the scientific object is `trans`."""

    def __init__(self, obs_len: int = 39, d: int = 512):
        super().__init__()
        self.obs_len = obs_len
        self.enc_tok = nn.Embedding(256, 32)
        self.enc_cell = nn.Linear(32, N_CELL)   # shared per-position head: exact copy generalization
        self.enc_flag = nn.Linear(32, N_FLAG)
        self.act_emb = nn.Embedding(N_ACT, 32)
        # Conv-equivariant transition: grid dynamics are LOCAL rules (agent moves
        # into the adjacent free cell, landmarks persist) — a flat MLP memorized
        # per-position statistics of the training layouts instead (Exp-#021 probe:
        # exact_oracle_d1 = 0% on held-out maps, all landmarks erased).
        cin = N_CELL + N_FLAG + 32
        self.trunk = nn.Sequential(
            nn.Conv2d(cin, 128, 3, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(),
        )
        self.head_cells = nn.Conv2d(128, N_CELL, 1)
        self.head_flag = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, N_FLAG))
        self.reward_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.trap_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def encode(self, idx: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """obs tokens -> state logits [B, GRID*N_CELL + N_FLAG]; optionally hard one-hot."""
        emb = self.enc_tok(idx)
        cell_logits = self.enc_cell(emb[:, 3 : 3 + GRID])          # [B,36,8]
        flag_logits = self.enc_flag(emb[:, 2])                   # [B,4]
        logits = torch.cat([cell_logits.flatten(1), flag_logits], dim=-1)
        if not hard:
            return logits
        cells, flags = self.split_state(logits)
        ch = F.one_hot(cells.argmax(-1), N_CELL).flatten(1).float()
        fh = F.one_hot(flags.argmax(-1), N_FLAG).float()
        return torch.cat([ch, fh], dim=-1)

    @staticmethod
    def split_state(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, : GRID * N_CELL].view(-1, GRID, N_CELL), x[:, GRID * N_CELL :]

    def step(self, s: torch.Tensor, action: torch.Tensor, hard: bool = False) -> tuple:
        """one-hot state [B, STATE_DIM] + action [B] -> (logits or hard state, rew, trap)."""
        cells, flags = self.split_state(s)                     # [B,36,8], [B,4]
        g = cells.transpose(1, 2).reshape(-1, N_CELL, 6, 6)    # [B,8,6,6]
        B = s.shape[0]
        fx = flags.unsqueeze(-1).unsqueeze(-1).expand(B, N_FLAG, 6, 6)
        ax = self.act_emb(action).unsqueeze(-1).unsqueeze(-1).expand(B, 32, 6, 6)
        h = self.trunk(torch.cat([g, fx, ax], dim=1))          # [B,128,6,6]
        cell_logits = self.head_cells(h).flatten(2).transpose(1, 2)  # [B,36,8]
        pooled = h.mean(dim=(2, 3))                            # [B,128]
        flag_logits = self.head_flag(pooled)                   # [B,4]
        rew = self.reward_head(pooled).squeeze(-1)
        trap = self.trap_head(pooled).squeeze(-1)
        logits = torch.cat([cell_logits.flatten(1), flag_logits], dim=-1)
        if not hard:
            return logits, rew, trap
        ch = F.one_hot(cell_logits.argmax(-1), N_CELL).flatten(1).float()
        fh = F.one_hot(flag_logits.argmax(-1), N_FLAG).float()
        return torch.cat([ch, fh], dim=-1), rew, trap


def decode_state(s: torch.Tensor) -> tuple[list[int], int]:
    """hard one-hot state -> (cells, flag)."""
    cells, flags = DiscreteBoardWM.split_state(s)
    return cells.argmax(-1)[0].tolist(), flags.argmax(-1).item()


@torch.no_grad()
def select_action_learned(wm: DiscreteBoardWM, obs_toks: list[int], horizon: int = 16, beam_width: int = 64, gamma: float = 0.97, device: str = "cpu") -> int:
    """Fully-learned beam-search MPC: exact symbolic search structure (Exp #018)
    but the transition is the LEARNED conv dynamics T(s,a) — no hand-coded rules.
    Node = (hard_state, score, first_action, terminal). Terminal = trapped (the
    '6' cell is gone — agent stands on it) or solved (flag bit1 && '5' gone)."""
    x = torch.tensor([obs_toks], dtype=torch.long, device=device)
    s0 = wm.encode(x, hard=True)
    nodes = [(s0, 0.0, -1, False)]
    best_first, best_score = 0, -1e9
    for h in range(horizon):
        cands = []
        for st, sc, first_a, term in nodes:
            if term:
                continue
            for a in range(4):
                sn, r, tr = wm.step(st, torch.tensor([a], device=device), hard=True)
                p_trap = torch.sigmoid(tr).item()
                cells, flag = decode_state(sn)
                hk, do = flag & 1, (flag >> 1) & 1
                trapped = 6 not in cells
                solved = do == 1 and 5 not in cells
                nsc = sc + (gamma**h) * (r.item() - 4.0 * p_trap)
                fa = a if h == 0 else first_a
                score = nsc
                if solved:
                    score += 10.0
                elif trapped:
                    score -= 5.0
                else:
                    ag = cells.index(2) if 2 in cells else -1
                    key_pos = cells.index(3) if 3 in cells else -1
                    door_pos = cells.index(4) if 4 in cells else -1
                    goal_pos = cells.index(5) if 5 in cells else -1
                    tgt = key_pos if not hk else (door_pos if not do else goal_pos)
                    blocked = set() if hk else {4}
                    d = _bfs(cells, ag, blocked)
                    score -= 0.06 * (d[tgt] if 0 <= tgt < len(d) and d[tgt] >= 0 else 12)
                cands.append((score, sn, nsc, fa, trapped or solved))
        cands.sort(key=lambda t: -t[0])
        seen, nodes = set(), []
        for score, sn, nsc, fa, term in cands:
            cells, flag = decode_state(sn)
            ag = cells.index(2) if 2 in cells else -1
            key = (ag, flag)
            if key in seen:
                continue
            seen.add(key)
            nodes.append((sn, nsc, fa, term))
            if score > best_score and fa >= 0:
                best_score, best_first = score, fa
            if len(nodes) >= beam_width:
                break
        if not nodes:
            break
    return best_first


@torch.no_grad()
def eval_learned_planning(wm: DiscreteBoardWM, n_maps: int, device: str, seed0: int = 50000, horizon: int = 16, max_steps: int = 18) -> dict:
    """Held-out planning with the FULLY-LEARNED pipeline: obs -> encode -> learned-T
    beam search -> env step. Same protocol as evaluate_planning's symbolic modes."""
    wm.eval()
    goals = traps = steps_sum = 0
    rets = []
    for m in range(n_maps):
        env = KeyDoorTrapEnv(seed0 + m)
        obs = env.reset()
        ret, n_steps = 0.0, 0
        while not env.done and n_steps < max_steps:
            a = select_action_learned(wm, obs[0], horizon=horizon, device=device)
            obs, r, done, trapped = env.step(a)
            ret += r
            n_steps += 1
        goals += int(env.agent == env.goal and env.door_open)
        traps += int(env.trapped)
        steps_sum += n_steps
        rets.append(ret)
    return {
        "goal_rate": round(goals / n_maps, 4),
        "trap_rate": round(traps / n_maps, 4),
        "mean_return": round(sum(rets) / n_maps, 4),
        "mean_steps": round(steps_sum / n_maps, 2),
    }


def batch_target(batch, device):
    """Parsed next-state one-hot targets + scalar targets from a transition batch."""
    cells_t, flags_t = [], []
    for b in batch:
        c, f = parse_state(b["next_obs"][0])
        cells_t.append(c)
        flags_t.append(f)
    cells = torch.tensor(cells_t, dtype=torch.long, device=device)  # [B,36]
    flags = torch.tensor(flags_t, dtype=torch.long, device=device)  # [B]
    return cells, flags


def onehot_from_parsed(cells: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    ch = F.one_hot(cells.clamp(0, N_CELL - 1), N_CELL).flatten(1).float()
    fh = F.one_hot(flags.clamp(0, N_FLAG - 1), N_FLAG).float()
    return torch.cat([ch, fh], dim=-1)


def train(wm: DiscreteBoardWM, transitions: list[dict], steps: int, bs: int, device: str, seed: int = 42):
    opt = torch.optim.AdamW(wm.parameters(), lr=2e-3, weight_decay=1e-2)
    rng = random.Random(seed)
    wm.train()
    # same-episode predecessor map for real scheduled sampling (consecutive entries
    # within an episode are consecutive env steps)
    prev_of = {i: i - 1 for i in range(1, len(transitions)) if transitions[i]["ep"] == transitions[i - 1]["ep"]}
    for it in range(steps):
        idxs = rng.sample(range(len(transitions)), bs)
        batch = [transitions[i] for i in idxs]
        x_cur = torch.tensor([b["obs"][0] for b in batch], dtype=torch.long, device=device)
        acts = torch.tensor([b["act"] for b in batch], dtype=torch.long, device=device)
        rews = torch.tensor([b["rew"] for b in batch], dtype=torch.float32, device=device)
        traps = torch.tensor([b["trapped"] for b in batch], dtype=torch.float32, device=device)

        cells_cur = torch.tensor([parse_state(b["obs"][0])[0] for b in batch], dtype=torch.long, device=device)
        flags_cur = torch.tensor([parse_state(b["obs"][0])[1] for b in batch], dtype=torch.long, device=device)
        cells_nxt, flags_nxt = batch_target(batch, device)
        s_true = onehot_from_parsed(cells_cur, flags_cur)
        # Scheduled sampling: with ramping probability feed trans the model's OWN
        # one-step prediction of s_t (from the same-episode predecessor step), so it
        # trains on the imperfect-state distribution it faces at rollout (Exp-#021
        # follow-up: discreteness alone didn't fix rollouts — trans trained only on
        # perfect states diverged on its own errors).
        ss = min(0.9, it / (0.5 * steps))
        if rng.random() < ss:
            s_in = s_true.clone()
            for j, i in enumerate(idxs):
                p = prev_of.get(i)
                if p is None:
                    continue
                tp = transitions[p]
                c_p, f_p = parse_state(tp["obs"][0])
                s_p = onehot_from_parsed(
                    torch.tensor([c_p], dtype=torch.long, device=device),
                    torch.tensor([f_p], dtype=torch.long, device=device),
                )
                a_p = torch.tensor([tp["act"]], dtype=torch.long, device=device)
                with torch.no_grad():
                    s_in[j : j + 1], _, _ = wm.step(s_p, a_p, hard=True)
        else:
            s_in = s_true

        enc_logits = wm.encode(x_cur)
        enc_cells, enc_flags = wm.split_state(enc_logits)
        loss_enc = F.cross_entropy(enc_cells.reshape(-1, N_CELL), cells_cur.clamp(0, N_CELL - 1).reshape(-1))
        loss_enc = loss_enc + F.cross_entropy(enc_flags, flags_cur.clamp(0, N_FLAG - 1))

        logits_nxt, r_pred, trap_logit = wm.step(s_in, acts)
        pred_cells, pred_flags = wm.split_state(logits_nxt)
        loss_tr = F.cross_entropy(pred_cells.reshape(-1, N_CELL), cells_nxt.clamp(0, N_CELL - 1).reshape(-1))
        loss_tr = loss_tr + F.cross_entropy(pred_flags, flags_nxt.clamp(0, N_FLAG - 1))
        loss_r = F.mse_loss(r_pred, rews)
        loss_t = F.binary_cross_entropy_with_logits(trap_logit, traps)
        loss = loss_enc + loss_tr + 3.0 * loss_r + 3.0 * loss_t

        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 500 == 0:
            with torch.no_grad():
                acc_cell = (enc_cells.argmax(-1) == cells_cur.clamp(0, N_CELL - 1)).float().mean().item()
                acc_tr = (pred_cells.argmax(-1) == cells_nxt.clamp(0, N_CELL - 1)).float().mean().item()
                acc_fl = (pred_flags.argmax(-1) == flags_nxt.clamp(0, N_FLAG - 1)).float().mean().item()
            print(
                f"step {it+1}/{steps} loss {loss.item():.3f} | enc_cell {acc_cell:.4f} trans_cell {acc_tr:.4f} trans_flag {acc_fl:.4f}",
                flush=True,
            )


@torch.no_grad()
def eval_rollout(wm: DiscreteBoardWM, n_maps: int, depth: int, device: str, seed0: int = 50000) -> dict:
    """Imagine from the ENCODED initial obs (full neural pipeline) for `depth` steps;
    score per-step fidelity against ground-truth parsed states."""
    wm.eval()
    stats = {k: [[] for _ in range(depth)] for k in ["exact", "agent", "landmark", "exact_oracle"]}
    for m in range(n_maps):
        env = KeyDoorTrapEnv(seed0 + m)
        obs = env.reset()
        plan = list(env.optimal_actions)
        x = torch.tensor([obs[0]], dtype=torch.long, device=device)
        s = wm.encode(x, hard=True)  # discrete encoded state
        s_or = None  # oracle-discrete track: parsed true state -> trans (encoder-free)
        for t in range(depth):
            if env.done or t >= len(plan):
                break
            a = torch.tensor([plan[t]], dtype=torch.long, device=device)
            s, _, _ = wm.step(s, a, hard=True)
            if t == 0:
                c0, f0 = parse_state(obs[0])
                s_or = onehot_from_parsed(
                    torch.tensor([c0], dtype=torch.long, device=device),
                    torch.tensor([f0], dtype=torch.long, device=device),
                )
            s_or, _, _ = wm.step(s_or, a, hard=True)
            obs, rew, done, trapped = env.step(int(plan[t]))
            tc, tf = parse_state(obs[0])
            pc, pf = wm.split_state(s)
            pred_cells = pc.argmax(-1)[0].tolist()
            pred_flag = pf.argmax(-1).item()
            true_cells = [min(max(int(c), 0), N_CELL - 1) for c in tc]
            stats["exact"][t].append(float(pred_cells == true_cells and pred_flag == tf))
            stats["agent"][t].append(float(2 in pred_cells and true_cells.index(2) == pred_cells.index(2)))
            lm_true = {true_cells.index(v) for v in (3, 4, 5, 6) if v in true_cells}
            lm_pred = {pred_cells.index(v) for v in (3, 4, 5, 6) if v in pred_cells}
            stats["landmark"][t].append(float(lm_true == lm_pred))
            oc, of = wm.split_state(s_or)
            oc_list = oc.argmax(-1)[0].tolist()
            stats["exact_oracle"][t].append(float(oc_list == true_cells and of.argmax(-1).item() == tf))
    out = {}
    for t in range(depth):
        for k in stats:
            if stats[k][t]:
                out[f"{k}_d{t+1}"] = round(sum(stats[k][t]) / len(stats[k][t]), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--n-envs", type=int, default=600)
    ap.add_argument("--eval-depth", type=int, default=14)
    ap.add_argument("--eval-maps", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="cortex_runs/discrete_wm_eval.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"[discrete_wm] collecting {args.n_envs} env trajectories...", flush=True)
    transitions = collect_world_model_dataset(n_envs=args.n_envs, seed_start=10000)
    print(f"[discrete_wm] {len(transitions)} transitions; training {args.steps} steps on {args.device}", flush=True)
    wm = DiscreteBoardWM().to(args.device)
    train(wm, transitions, args.steps, args.bs, args.device, args.seed)
    torch.save(wm.state_dict(), "cortex_runs/discrete_wm.pt")

    print(f"[discrete_wm] rollout-fidelity eval: {args.eval_maps} held-out maps, depth {args.eval_depth}", flush=True)
    res = eval_rollout(wm, args.eval_maps, args.eval_depth, args.device)
    res["steps"] = args.steps
    res["n_transitions"] = len(transitions)
    print(json.dumps(res, indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
