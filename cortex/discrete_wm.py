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
SS_DEPTH = 4  # max scheduled-sampling chain length (model unrolls up to 4 steps back)


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


def extract_statics(cells: list[int]) -> dict:
    """Immovable per-episode structure: walls + landmark positions. Landmarks never
    move in this env — they are only consumed (flag flips) or overlaid by the agent."""
    return {
        "walls": [i for i, c in enumerate(cells) if c == 1],
        "key": cells.index(3) if 3 in cells else -1,
        "door": cells.index(4) if 4 in cells else -1,
        "goal": cells.index(5) if 5 in cells else -1,
        "trap": cells.index(6) if 6 in cells else -1,
    }


def render_factored(agent: int, flag: int, st: dict) -> list[int]:
    """(agent_pos, flag, statics) -> board, same overlay order as KeyDoorTrapEnv.obs()."""
    g = [0] * GRID
    for w in st["walls"]:
        g[w] = 1
    if st["trap"] >= 0:
        g[st["trap"]] = 6
    if not (flag & 1) and st["key"] >= 0:
        g[st["key"]] = 3
    if not (flag & 2) and st["door"] >= 0:
        g[st["door"]] = 4
    if st["goal"] >= 0:
        g[st["goal"]] = 5
    if agent >= 0:
        g[agent] = 2
    return g


def surgical_step(wm: DiscreteBoardWM, s: torch.Tensor, a: torch.Tensor):
    """Minimal manifold projection: per-cell argmax for everything, then enforce the
    one global constraint per-cell CE cannot express — exactly one agent. Absent ->
    revive at argmax over cell-2 logits; multiple -> keep highest logit-2 cell, demote
    others to their 2nd-best class. Statics/landmarks untouched (Exp-#025 showed the
    residual failure is agent evaporation ~25% of steps, not landmark corruption)."""
    logits, rew, tr = wm.step(s, a)
    cl, fl = wm.split_state(logits)
    cl0, fl0 = cl[0].clone(), fl[0]
    cells = cl0.argmax(-1).tolist()
    ag_pos = [i for i, c in enumerate(cells) if c == 2]
    if len(ag_pos) == 0:
        cells[int(cl0[:, 2].argmax().item())] = 2
    elif len(ag_pos) > 1:
        keep = max(ag_pos, key=lambda i: cl0[i, 2].item())
        for i in ag_pos:
            if i != keep:
                cl0[i, 2] = -1e9
                cells[i] = int(cl0[i].argmax().item())
    flag = int(fl0.argmax().item())
    s2 = onehot_from_parsed(
        torch.tensor([cells], dtype=torch.long, device=s.device),
        torch.tensor([flag], dtype=torch.long, device=s.device),
    )
    return s2, rew, tr


def factored_step(wm: DiscreteBoardWM, s: torch.Tensor, a: torch.Tensor, st: dict, learn_flag: bool = True):
    """Projected rollout step: the conv net predicts only what can CHANGE — the next
    agent position (argmax over cell-2 logits, walls masked) and optionally the flag.
    With learn_flag=False the flag is derived from the milestone rules (agent-on-key
    -> has_key; agent-on-door with key -> door_open), reducing the learned output to
    a single moving part. Locked-door entries revert to the previous agent position
    (env: blocked move). The board is then re-rendered exactly — object permanence as
    a hard constraint instead of re-predicting 36 independent cells every step."""
    logits, rew, tr = wm.step(s, a)
    cl, fl = wm.split_state(logits)
    prev_cells, prev_flag = decode_state(s)
    prev_agent = prev_cells.index(2) if 2 in prev_cells else -1
    alog = cl[0, :, 2].clone()
    alog[st["walls"]] = -1e9
    agent = int(alog.argmax().item())
    flag = int(fl[0].argmax().item()) if learn_flag else prev_flag
    if agent == st["door"] and not (flag & 1):
        agent = prev_agent  # env: locked door blocks the move
    if not learn_flag:
        if agent == st["key"] >= 0:
            flag |= 1
        elif agent == st["door"] >= 0 and (flag & 1):
            flag |= 2
    board = render_factored(agent, flag, st)
    s2 = onehot_from_parsed(
        torch.tensor([board], dtype=torch.long, device=s.device),
        torch.tensor([flag], dtype=torch.long, device=s.device),
    )
    return s2, rew, tr, agent, flag


class ValueNet(nn.Module):
    """V(s) -> BFS distance to the current milestone target, trained on TRUE parsed
    transitions. Beam leaves get a learned potential instead of _bfs on possibly-
    corrupt imagined boards (Exp #027: scoring/dedup/selection/search-budget all
    cleared as the residual; the flat-horizon leaf value is what remains)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(N_CELL + N_FLAG, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 36, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        cells, flags = DiscreteBoardWM.split_state(s)
        B = s.shape[0]
        x = torch.cat(
            [
                cells.transpose(1, 2).reshape(B, N_CELL, 6, 6),
                flags.unsqueeze(-1).unsqueeze(-1).expand(B, N_FLAG, 6, 6),
            ],
            1,
        )
        return self.net(x).squeeze(-1)


def milestone_dist(cells: list[int], flag: int) -> float:
    """Env-exact steps to the CURRENT milestone target (key -> door -> goal)."""
    hk, do = flag & 1, (flag >> 1) & 1
    if 2 not in cells:
        return 12.0
    ag = cells.index(2)
    tgt = (
        cells.index(3) if (not hk and 3 in cells)
        else cells.index(4) if (not do and 4 in cells)
        else cells.index(5) if 5 in cells
        else ag
    )
    d = _bfs(cells, ag, set() if hk else {4})
    v = d[tgt] if 0 <= tgt < len(d) and d[tgt] >= 0 else 12
    return float(min(v, 12))


def train_value(vn: ValueNet, transitions: list[dict], steps: int = 1500, bs: int = 256, device: str = "cuda", seed: int = 0) -> ValueNet:
    rng = random.Random(seed)
    vn.to(device).train()
    opt = torch.optim.AdamW(vn.parameters(), lr=3e-4)
    for it in range(steps):
        batch = rng.sample(transitions, bs)
        with torch.no_grad():
            ss, yy = [], []
            for b in batch:
                c, f = parse_state(b["obs"][0])
                ss.append(c)
                yy.append(milestone_dist(c, f))
            cells_b = torch.tensor(ss, dtype=torch.long, device=device)
            flags_b = torch.tensor([f for _, f in (parse_state(b["obs"][0]) for b in batch)], dtype=torch.long, device=device)
            s = onehot_from_parsed(cells_b, flags_b)
            y = torch.tensor(yy, device=device)
        loss = F.mse_loss(vn(s), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 300 == 0:
            print(f"[value] step {it+1} loss {loss.item():.4f}", flush=True)
    vn.eval()
    return vn


def select_action_learned(wm: DiscreteBoardWM, obs_toks: list[int], horizon: int = 16, beam_width: int = 64, gamma: float = 0.97, device: str = "cpu", mode: str = "raw", vn: ValueNet | None = None) -> int:
    """Fully-learned beam-search MPC: exact symbolic search structure (Exp #018)
    but the transition is the LEARNED conv dynamics T(s,a) — no hand-coded rules.
    Node = (hard_state, score, first_action, terminal). Terminal = trapped (the
    '6' cell is gone — agent stands on it) or solved (flag bit1 && '5' gone).
    mode 'raw': per-cell argmax rollout; 'surg': argmax + one-agent constraint only;
    'pin': factored step, learned flag; 'fact': factored step, derived flag;
    'staged': raw + popcount(flag) milestone-progress bonus (Exp-#025: failures are
    stage-transition stalls — never reaching the flag flip, not mid-path drift)."""
    x = torch.tensor([obs_toks], dtype=torch.long, device=device)
    s0 = wm.encode(x, hard=True)
    st0 = extract_statics(decode_state(s0)[0]) if mode in ("pin", "fact") else None
    nodes = [(s0, 0.0, -1, False)]
    best_first, best_score = 0, -1e9
    for h in range(horizon):
        cands = []
        for st, sc, first_a, term in nodes:
            if term:
                continue
            for a in range(4):
                if mode in ("raw", "staged", "agflag", "val"):
                    sn, r, tr = wm.step(st, torch.tensor([a], device=device), hard=True)
                elif mode == "surg":
                    sn, r, tr = surgical_step(wm, st, torch.tensor([a], device=device))
                else:
                    sn, r, tr, _, _ = factored_step(wm, st, torch.tensor([a], device=device), st0, learn_flag=(mode == "pin"))
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
                    if mode == "staged":
                        score += 3.0 * bin(flag).count("1")
                    if mode == "val" and vn is not None:
                        score -= 0.3 * vn(sn).item()
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
            # symbolic-planner dedup used (ag, flag) — correct there because that IS the
            # whole state, but two DISTINCT imagined boards sharing (ag,flag) are not
            # equivalent; dedup on the full board keeps beam diversity (Exp #026).
            key = (ag, flag) if mode == "agflag" else tuple(cells) + (flag,)
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
def eval_learned_planning(wm: DiscreteBoardWM, n_maps: int, device: str, seed0: int = 50000, horizon: int = 16, max_steps: int = 18, mode: str = "raw", vn: ValueNet | None = None) -> dict:
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
            a = select_action_learned(wm, obs[0], horizon=horizon, device=device, mode=mode, vn=vn)
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


def train(wm: DiscreteBoardWM, transitions: list[dict], steps: int, bs: int, device: str, seed: int = 42, aux_agent: bool = True, ss_depth: int = SS_DEPTH, ckpt_every: int = 0, ckpt_prefix: str | None = None):
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
        # Scheduled sampling: with ramping probability feed trans its OWN n-step
        # prediction of s_t (walk the same-episode predecessor chain up to SS_DEPTH
        # steps back through the model itself), exposing it to the imperfect-state
        # distribution it faces at rollout (Exp-#021/#023 lineage).
        ss = min(0.9, it / (0.5 * steps))
        if rng.random() < ss:
            s_in = s_true.clone()
            for j, i in enumerate(idxs):
                p = prev_of.get(i)
                if p is None:
                    continue
                depth = 1 + rng.randrange(ss_depth)
                chain = [i]
                while len(chain) <= depth and prev_of.get(chain[-1]) is not None:
                    chain.append(prev_of[chain[-1]])
                if len(chain) <= 1:
                    continue
                start = transitions[chain[-1]]
                c_p, f_p = parse_state(start["obs"][0])
                s_p = onehot_from_parsed(
                    torch.tensor([c_p], dtype=torch.long, device=device),
                    torch.tensor([f_p], dtype=torch.long, device=device),
                )
                with torch.no_grad():
                    for ci in reversed(chain[1:]):
                        a_p = torch.tensor([transitions[ci]["act"]], dtype=torch.long, device=device)
                        s_p, _, _ = wm.step(s_p, a_p, hard=True)
                s_in[j : j + 1] = s_p
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
        # Agent-existence auxiliary: the dominant v6 residual was the agent
        # evaporating from imagined boards (~25% of rollout steps; per-cell CE pays
        # only ~1/36 for it). Penalize the SOFT agent count deviating from exactly 1 —
        # under scheduled sampling this teaches self-correction from degraded inputs.
        cnt = pred_cells.softmax(-1)[..., 2].sum(-1)
        loss_cnt = F.mse_loss(cnt, torch.ones_like(cnt)) if aux_agent else torch.zeros((), device=device)
        loss_r = F.mse_loss(r_pred, rews)
        loss_t = F.binary_cross_entropy_with_logits(trap_logit, traps)
        loss = loss_enc + loss_tr + 3.0 * loss_r + 3.0 * loss_t + loss_cnt

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
        # Periodic checkpointing: Exp #026 showed teacher-forced metrics are blind to
        # rollout robustness — same recipe can land in a brittle basin (v8) or a
        # robust one (v6). Cheap insurance against late-training instability.
        if ckpt_every and ckpt_prefix and (it + 1) % ckpt_every == 0:
            torch.save(wm.state_dict(), f"{ckpt_prefix}_step{it+1}.pt")


@torch.no_grad()
def eval_rollout(wm: DiscreteBoardWM, n_maps: int, depth: int, device: str, seed0: int = 50000) -> dict:
    """Imagine from the ENCODED initial obs (full neural pipeline) for `depth` steps;
    score per-step fidelity against ground-truth parsed states."""
    wm.eval()
    tracks = ["exact", "agent", "landmark", "exact_oracle", "exact_fact", "agent_fact", "landmark_fact"]
    stats = {k: [[] for _ in range(depth)] for k in tracks}
    for m in range(n_maps):
        env = KeyDoorTrapEnv(seed0 + m)
        obs = env.reset()
        plan = list(env.optimal_actions)
        x = torch.tensor([obs[0]], dtype=torch.long, device=device)
        s = wm.encode(x, hard=True)  # discrete encoded state
        s_fact = s.clone()
        st0 = extract_statics(decode_state(s)[0])
        s_or = None  # oracle-discrete track: parsed true state -> trans (encoder-free)
        for t in range(depth):
            if env.done or t >= len(plan):
                break
            a = torch.tensor([plan[t]], dtype=torch.long, device=device)
            s, _, _ = wm.step(s, a, hard=True)
            s_fact, _, _, _, _ = factored_step(wm, s_fact, a, st0, learn_flag=False)
            if t == 0:
                c0, f0 = parse_state(obs[0])
                s_or = onehot_from_parsed(
                    torch.tensor([c0], dtype=torch.long, device=device),
                    torch.tensor([f0], dtype=torch.long, device=device),
                )
            s_or, _, _ = wm.step(s_or, a, hard=True)
            obs, rew, done, trapped = env.step(int(plan[t]))
            tc, tf = parse_state(obs[0])
            true_cells = [min(max(int(c), 0), N_CELL - 1) for c in tc]
            lm_true = {true_cells.index(v) for v in (3, 4, 5, 6) if v in true_cells}
            for tag, s_pred in (("", s), ("_fact", s_fact)):
                p_cells, p_flag = decode_state(s_pred)
                stats[f"exact{tag}"][t].append(float(p_cells == true_cells and p_flag == tf))
                stats[f"agent{tag}"][t].append(float(2 in p_cells and true_cells.index(2) == p_cells.index(2)))
                lm_pred = {p_cells.index(v) for v in (3, 4, 5, 6) if v in p_cells}
                stats[f"landmark{tag}"][t].append(float(lm_true == lm_pred))
            oc, of = wm.split_state(s_or)
            oc_list = oc.argmax(-1)[0].tolist()
            stats["exact_oracle"][t].append(float(oc_list == true_cells and of.argmax(-1).item() == tf))
    out = {}
    for t in range(depth):
        for k in stats:
            if stats[k][t]:
                out[f"{k}_d{t+1}"] = round(sum(stats[k][t]) / len(stats[k][t]), 4)
    return out


@torch.no_grad()
def select_checkpoint(wm: DiscreteBoardWM, paths: list[str], device: str, n_maps: int = 16, depth: int = 8, metric: str = "fidelity", sel_seed0: int = 70000) -> str:
    """Model selection on the claimed metric. Exp #026 showed fidelity-selection
    ANTI-selects for planning (mid-training peak vs monotone planning growth) —
    'plan' scores each ckpt by goal rate on a disjoint seed range."""
    best_path, best_score = paths[-1], -1.0
    for p in paths:
        wm.load_state_dict(torch.load(p, map_location=device))
        if metric == "plan":
            r = eval_learned_planning(wm, n_maps, device, seed0=sel_seed0)
            sc = r["goal_rate"]
        else:
            r = eval_rollout(wm, n_maps, depth, device)
            tail = [v for k, v in r.items() if k.startswith(("exact_d", "agent_d"))]
            sc = sum(tail) / len(tail)
        print(f"[select:{metric}] {p}: score {sc:.3f}", flush=True)
        if sc > best_score:
            best_score, best_path = sc, p
    print(f"[select:{metric}] chose {best_path}", flush=True)
    wm.load_state_dict(torch.load(best_path, map_location=device))
    return best_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--n-envs", type=int, default=600)
    ap.add_argument("--eval-depth", type=int, default=14)
    ap.add_argument("--eval-maps", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load", default=None, help="checkpoint to load; skips training")
    ap.add_argument("--ckpt-out", default="cortex_runs/discrete_wm.pt")
    ap.add_argument("--aux-agent", type=int, default=1, help="agent-existence auxiliary loss")
    ap.add_argument("--ss-depth", type=int, default=SS_DEPTH)
    ap.add_argument("--ckpt-every", type=int, default=0, help="save ckpt every N steps for fidelity selection")
    ap.add_argument("--select-metric", default="fidelity", choices=["fidelity", "plan"])
    ap.add_argument("--plan-modes", default=None, help="comma list: run eval_learned_planning per mode")
    ap.add_argument("--out", default="cortex_runs/discrete_wm_eval.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    wm = DiscreteBoardWM().to(args.device)
    if args.load:
        wm.load_state_dict(torch.load(args.load, map_location=args.device))
        print(f"[discrete_wm] loaded {args.load}", flush=True)
    else:
        print(f"[discrete_wm] collecting {args.n_envs} env trajectories...", flush=True)
        transitions = collect_world_model_dataset(n_envs=args.n_envs, seed_start=10000)
        print(f"[discrete_wm] {len(transitions)} transitions; training {args.steps} steps on {args.device}", flush=True)
        train(wm, transitions, args.steps, args.bs, args.device, args.seed, aux_agent=bool(args.aux_agent), ss_depth=args.ss_depth,
              ckpt_every=args.ckpt_every, ckpt_prefix=args.ckpt_out.replace(".pt", ""))
        torch.save(wm.state_dict(), args.ckpt_out)
        print(f"[discrete_wm] saved {args.ckpt_out}", flush=True)
        if args.ckpt_every:
            import glob as _g
            paths = sorted(_g.glob(args.ckpt_out.replace(".pt", "_step*.pt")), key=lambda p: int(p.split("_step")[-1][:-3]))
            paths.append(args.ckpt_out)
            best = select_checkpoint(wm, paths, args.device, metric=args.select_metric)
            wm.load_state_dict(torch.load(best, map_location=args.device))
            torch.save(wm.state_dict(), args.ckpt_out)  # selected -> canonical ckpt

    print(f"[discrete_wm] rollout-fidelity eval: {args.eval_maps} held-out maps, depth {args.eval_depth}", flush=True)
    res = eval_rollout(wm, args.eval_maps, args.eval_depth, args.device)
    res["steps"] = args.steps
    if not args.load:
        res["n_transitions"] = len(transitions)
    if args.plan_modes:
        for mode in args.plan_modes.split(","):
            pres = eval_learned_planning(wm, args.eval_maps, args.device, mode=mode)
            res[f"plan_{mode}"] = pres
            print(f"[discrete_wm] plan {mode}: {pres}", flush=True)
    print(json.dumps(res, indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
