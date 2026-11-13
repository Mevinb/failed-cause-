"""Axis 7: Embodied Key-Door-BaitTrap-Goal World-Model & Latent MPC Lookahead Benchmark."""
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from cortex.tasks import COLOR_BASE, TOK_IN_GRID, TOK_RESET

# Cell types in 6x6 grid:
# 0: empty, 1: wall, 2: agent, 3: key, 4: locked_door, 5: goal (+2.0), 6: bait_trap (+0.5 then permanent trap!)
ACTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]  # N, S, E, W, Stay


class KeyDoorTrapEnv:
    """Procedural 6x6 gridworld with a tempting +0.5 adjacent Bait Trap vs +2.5 Key->Door->Goal optimal path."""

    size = 6

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        S = self.size
        while True:
            self.walls = set()
            for _ in range(5):
                self.walls.add((self.rng.randrange(S), self.rng.randrange(S)))
            free = [(r, c) for r in range(S) for c in range(S) if (r, c) not in self.walls]
            if len(free) < 10:
                continue
            self.agent = self.rng.choice(free)
            # Place bait trap adjacent to start so greedy 1-step policies get lured immediately!
            adj = [
                (self.agent[0] + dr, self.agent[1] + dc)
                for dr, dc in ACTIONS[:4]
                if 0 <= self.agent[0] + dr < S and 0 <= self.agent[1] + dc < S and (self.agent[0] + dr, self.agent[1] + dc) not in self.walls
            ]
            if len(adj) < 2:
                continue
            self.trap = adj[0]
            rem = [p for p in free if p not in (self.agent, self.trap)]
            self.key, self.door, self.goal = self.rng.sample(rem, 3)
            # Verify solvable path: agent -> key (avoiding trap & locked door) AND key -> door -> goal (avoiding trap)
            p1 = self._bfs(self.agent, self.key, blocked=self.walls | {self.trap, self.door})
            p2 = self._bfs(self.key, self.door, blocked=self.walls | {self.trap})
            p3 = self._bfs(self.door, self.goal, blocked=self.walls | {self.trap})
            if p1 is not None and p2 is not None and p3 is not None and len(p1) + len(p2) + len(p3) <= 14:
                self.optimal_actions = p1 + p2 + p3
                break
        self.has_key = False
        self.door_open = False
        self.trapped = False
        self.done = False
        self.steps = 0
        return self.obs()

    def _bfs(self, start, target, blocked):
        q = deque([(start, [])])
        seen = {start}
        while q:
            (r, c), path = q.popleft()
            if (r, c) == target:
                return path
            for a_idx, (dr, dc) in enumerate(ACTIONS[:4]):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in blocked and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append(((nr, nc), path + [a_idx]))
        return None

    def obs(self) -> tuple[list[int], list[int], list[int]]:
        S = self.size
        g = np.zeros((S, S), dtype=np.int64)
        for r, c in self.walls:
            g[r, c] = 1
        g[self.trap] = 6
        if not self.has_key:
            g[self.key] = 3
        if not self.door_open:
            g[self.door] = 4
        g[self.goal] = 5
        g[self.agent] = 2
        toks = [TOK_RESET, TOK_IN_GRID, COLOR_BASE + int(self.has_key) + 2 * int(self.door_open)]
        rows = [0, 0, 0]
        cols = [0, 0, 0]
        for r in range(S):
            for c in range(S):
                toks.append(COLOR_BASE + int(g[r, c]))
                rows.append(r + 1)
                cols.append(c + 1)
        return toks, rows, cols

    def step(self, action: int) -> tuple[tuple[list[int], list[int], list[int]], float, bool, bool]:
        self.steps += 1
        if self.done:
            return self.obs(), 0.0, True, self.trapped
        if self.trapped:
            if self.steps >= 18:
                self.done = True
            return self.obs(), -0.05, self.done, True

        dr, dc = ACTIONS[action]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc
        rew = -0.01
        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.walls:
            if (nr, nc) == self.door and not self.has_key:
                pass  # locked door blocks movement
            else:
                self.agent = (nr, nc)
                if self.agent == self.trap:
                    rew = 0.50
                    self.trapped = True
                elif self.agent == self.key and not self.has_key:
                    self.has_key = True
                    rew = 0.30
                elif self.agent == self.door and self.has_key and not self.door_open:
                    self.door_open = True
                    rew = 0.30
                elif self.agent == self.goal and self.door_open:
                    rew = 2.00
                    self.done = True
        if self.steps >= 18:
            self.done = True
        return self.obs(), rew, self.done, self.trapped


def encode_obs_batch(obs_list, device: str) -> dict[str, torch.Tensor]:
    x = torch.tensor([o[0] for o in obs_list], dtype=torch.long, device=device)
    r = torch.tensor([o[1] for o in obs_list], dtype=torch.long, device=device)
    c = torch.tensor([o[2] for o in obs_list], dtype=torch.long, device=device)
    return {"idx": x, "rows": r, "cols": c, "modality": 3}


def collect_world_model_dataset(n_envs: int = 500, seed_start: int = 10000) -> list[dict]:
    """Collect mixed exploratory, bait-trap, and expert trajectories for world-model & value learning."""
    transitions = []
    rng = random.Random(seed_start)
    for idx in range(n_envs):
        env = KeyDoorTrapEnv(seed_start + idx)
        obs = env.reset()
        mode = idx % 3  # 0: expert + noise, 1: random explorer, 2: trap-experiencing
        opt = list(env.optimal_actions)
        step_i = 0
        episode = []
        while not env.done:
            if mode == 0 and step_i < len(opt) and rng.random() < 0.85:
                a = opt[step_i]
                is_expert = True
            elif mode == 2 and step_i == 0:
                # Explicitly experience stepping into the adjacent bait trap
                a = next(
                    i for i, (dr, dc) in enumerate(ACTIONS[:4])
                    if (env.agent[0] + dr, env.agent[1] + dc) == env.trap
                )
                is_expert = False
            else:
                a = rng.randrange(4)
                is_expert = False
            nxt_obs, rew, done, trapped = env.step(a)
            episode.append({
                "obs": obs, "act": a, "next_obs": nxt_obs,
                "rew": rew, "trapped": float(trapped), "done": float(done),
                "expert_act": opt[step_i] if (mode == 0 and step_i < len(opt)) else -100,
                "ep": idx,
            })
            obs = nxt_obs
            step_i += 1
        # Compute discounted Monte-Carlo return for value head supervision
        G = 0.0
        for tr in reversed(episode):
            G = tr["rew"] + 0.95 * G
            tr["ret"] = G
        transitions.extend(episode)
    return transitions


def _milestone_flags(obs_toks) -> tuple[int, int]:
    flag = obs_toks[2] - COLOR_BASE
    return flag & 1, (flag >> 1) & 1


def train_world_model(model, transitions: list[dict], steps: int = 900, batch_size: int = 64, device: str = "cuda", unroll_prob: float = 0.25):
    """Train Subsystem 4 (LatentWorldModel: Grounded JEPA + reward + trap + value + policy + subgoal heads + 2-step unroll)."""
    wm = model.world_model
    opt = torch.optim.AdamW(wm.parameters(), lr=2e-3, weight_decay=1e-2)
    rng = random.Random(42)
    model.eval()
    wm.train()
    has_subgoal = hasattr(wm, "milestone_head")
    for it in range(steps):
        batch = rng.sample(transitions, batch_size)
        x_cur = torch.tensor([b["obs"][0] for b in batch], dtype=torch.long, device=device)
        x_nxt = torch.tensor([b["next_obs"][0] for b in batch], dtype=torch.long, device=device)
        z_cur = wm.encode_obs(x_cur)
        with torch.no_grad():
            z_tgt = wm.encode_obs(x_nxt)
        acts = torch.tensor([b["act"] for b in batch], dtype=torch.long, device=device)
        rews = torch.tensor([b["rew"] for b in batch], dtype=torch.float32, device=device)
        traps = torch.tensor([b["trapped"] for b in batch], dtype=torch.float32, device=device)
        rets = torch.tensor([b["ret"] for b in batch], dtype=torch.float32, device=device)
        exp_a = torch.tensor([b["expert_act"] for b in batch], dtype=torch.long, device=device)

        z_pred, r_pred, trap_logit = wm.step_latent(z_cur, acts)
        v_pred = wm.value_head(z_cur).squeeze(-1)
        pol_logits = wm.policy_head(z_cur)
        # Categorical board reconstruction grounds latent rollouts (DreamerV3 principle)
        nxt_cells = (x_nxt - COLOR_BASE).clamp(0, wm.n_cell_types - 1)
        rec_cur = F.cross_entropy(wm.obs_dec(z_cur).view(-1, wm.n_cell_types), (x_cur - COLOR_BASE).clamp(0, wm.n_cell_types - 1).view(-1))
        rec_nxt = F.cross_entropy(wm.obs_dec(z_pred).view(-1, wm.n_cell_types), nxt_cells.view(-1))

        loss_jepa = F.mse_loss(z_pred, z_tgt)
        loss_rew = F.mse_loss(r_pred, rews)
        loss_trap = F.binary_cross_entropy_with_logits(trap_logit, traps)
        loss_val = F.mse_loss(v_pred, rets)
        loss_pol = F.cross_entropy(pol_logits, exp_a, ignore_index=-100) if (exp_a != -100).any() else 0.0
        loss = loss_jepa + rec_cur + rec_nxt + 3.0 * loss_rew + 3.0 * loss_trap + 1.5 * loss_val + 1.5 * loss_pol

        if has_subgoal:
            miles = torch.tensor(
                [[*_milestone_flags(b["obs"][0]), b["done"]] for b in batch], dtype=torch.float32, device=device
            )
            prog = []
            for b in batch:
                hk0, dr0 = _milestone_flags(b["obs"][0])
                hk1, dr1 = _milestone_flags(b["next_obs"][0])
                prog.append([float(hk1 and not hk0), float(dr1 and not dr0), float(b["done"] and b["rew"] > 1.0)])
            prog = torch.tensor(prog, dtype=torch.float32, device=device)
            loss = loss + 2.0 * F.binary_cross_entropy_with_logits(wm.milestone_head(z_cur), miles)
            za = torch.cat([z_cur, wm.act_emb(acts)], dim=-1)
            loss = loss + 2.0 * F.binary_cross_entropy_with_logits(wm.progress_head(za), prog)

        # Gen-2: 2-step latent unroll so imagined rollouts stay accurate at depth 8+
        if unroll_prob > 0 and rng.random() < unroll_prob:
            idx2 = [i for i in rng.sample(range(len(transitions) - 1), batch_size) if transitions[i]["ep"] == transitions[i + 1]["ep"]]
            if idx2:
                b0 = [transitions[i] for i in idx2]
                b1 = [transitions[i + 1] for i in idx2]
                z0 = wm.encode_obs(torch.tensor([b["obs"][0] for b in b0], dtype=torch.long, device=device))
                a0 = torch.tensor([b["act"] for b in b0], dtype=torch.long, device=device)
                a1 = torch.tensor([b["act"] for b in b1], dtype=torch.long, device=device)
                with torch.no_grad():
                    z_t = wm.encode_obs(torch.tensor([b["next_obs"][0] for b in b1], dtype=torch.long, device=device))
                z1, _, _ = wm.step_latent(z0, a0)
                z2, r2, t2 = wm.step_latent(z1, a1)
                cells2 = (torch.tensor([b["next_obs"][0] for b in b1], dtype=torch.long, device=device) - COLOR_BASE).clamp(0, wm.n_cell_types - 1)
                loss = loss + F.mse_loss(z2, z_t) + F.cross_entropy(wm.obs_dec(z2).view(-1, wm.n_cell_types), cells2.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    wm.eval()


def _dedup_beams(z: torch.Tensor, score: torch.Tensor) -> list[int]:
    """Keep the best-scoring candidate per quantized latent bucket (state deduplication)."""
    keys = (z * 4).round().to(torch.int16).cpu().numpy().tobytes()
    keep, seen = [], set()
    order = score.argsort(descending=True).tolist()
    row = z.shape[1] * 2  # int16 = 2 bytes
    for i in order:
        kb = keys[i * row : (i + 1) * row]
        if kb in seen:
            continue
        seen.add(kb)
        keep.append(i)
    return keep


@torch.no_grad()
def select_action_mpc(
    model,
    z0: torch.Tensor,
    horizon: int = 6,
    beam_width: int = 24,
    gamma: float = 0.95,
    recent_z: list[torch.Tensor] | None = None,
    mode: str = "flat",
) -> int:
    """Subsystem 6: Latent Beam-Search MPC. mode='flat': dense return. mode='subgoal':
    hierarchical planning toward the next unachieved milestone (key -> door -> goal),
    scoring beams by predicted subgoal progress + a progress-value terminal heuristic."""
    wm = model.world_model
    if horizon == 0:
        return int(wm.policy_head(z0).argmax(dim=-1).item())
    if horizon == 1:
        acts = torch.arange(4, device=z0.device)
        _, r1, _ = wm.step_latent(z0.expand(4, -1), acts)
        return int(r1.argmax().item())

    subgoal = None
    if mode == "subgoal" and hasattr(wm, "milestone_head"):
        miles = torch.sigmoid(wm.milestone_head(z0)).squeeze(0)  # [has_key, door_open, done]
        subgoal = 0 if miles[0] < 0.5 else (1 if miles[1] < 0.5 else 2)

    beams_z = z0
    beams_ret = torch.zeros(1, device=z0.device)
    beams_disc = torch.ones(1, device=z0.device)
    beams_first_a = torch.zeros(1, dtype=torch.long, device=z0.device)
    hist_stack = torch.cat(recent_z[-6:], dim=0) if recent_z else None

    for h in range(horizon):
        B = beams_z.shape[0]
        z_rep = beams_z.repeat_interleave(4, dim=0)
        acts = torch.arange(4, device=z0.device).repeat(B)
        z_nxt, r_pred, trap_logit = wm.step_latent(z_rep, acts)
        p_trap = torch.sigmoid(trap_logit)
        loop_pen = 0.0
        if hist_stack is not None and h < 2:
            sim = F.cosine_similarity(z_nxt.unsqueeze(1), hist_stack.unsqueeze(0), dim=-1).max(dim=1).values
            loop_pen = 0.35 * F.relu(sim - 0.92)
        prior_bonus = 0.08 * F.log_softmax(wm.policy_head(z_rep), dim=-1).gather(1, acts.unsqueeze(1)).squeeze(1)
        step_val = r_pred - 4.0 * p_trap - loop_pen + prior_bonus
        if subgoal is not None:
            za = torch.cat([z_rep, wm.act_emb(acts)], dim=-1)
            prog_logit = wm.progress_head(za)[:, subgoal]
            step_val = step_val + 3.0 * torch.sigmoid(prog_logit) + 0.5 * prog_logit.clamp(-8.0, 0.0)
        ret_nxt = beams_ret.repeat_interleave(4) + beams_disc.repeat_interleave(4) * step_val
        first_a = acts if h == 0 else beams_first_a.repeat_interleave(4)
        term = (gamma ** (h + 1)) * wm.value_head(z_nxt).squeeze(-1) * (1.0 - p_trap)
        if subgoal is not None:
            za_all = torch.cat([z_nxt.repeat_interleave(4, dim=0), wm.act_emb(torch.arange(4, device=z0.device).repeat(z_nxt.shape[0]))], dim=-1)
            prog_next = torch.sigmoid(wm.progress_head(za_all)[:, subgoal]).view(z_nxt.shape[0], 4).max(dim=1).values
            term = term + (gamma ** (h + 1)) * 4.0 * prog_next * (1.0 - p_trap)
        score = ret_nxt + term
        keep_ix = _dedup_beams(z_nxt, score) if z_nxt.shape[0] > beam_width else score.argsort(descending=True).tolist()
        keep_ix = keep_ix[: max(beam_width, 4 * beam_width // 3)]
        sel = torch.tensor(keep_ix[:beam_width] if len(keep_ix) >= beam_width else keep_ix, device=z0.device)
        beams_z = z_nxt[sel]
        beams_ret = ret_nxt[sel]
        beams_disc = beams_disc.repeat_interleave(4)[sel] * gamma * (1.0 - p_trap[sel])
        beams_first_a = first_a[sel]

    return int(beams_first_a[0].item())


@torch.no_grad()
def evaluate_planning(model, n_maps: int = 80, seed_start: int = 50000, device: str = "cuda") -> dict:
    """Compare Myopic (H=1), Reactive Policy (H=0), and Latent World-Model MPC (H=6) on unseen maps."""
    results = {}
    modes = [("myopic_H1", 1, "flat"), ("reactive_policy_H0", 0, "flat"), ("world_model_mpc_H6", 6, "flat")]
    if hasattr(model.world_model, "milestone_head"):
        modes.append(("hierarchical_subgoal_H8", 8, "subgoal"))
    for mode_name, horizon, mode in modes:
        solved, trapped, returns = 0, 0, []
        for i in range(n_maps):
            env = KeyDoorTrapEnv(seed_start + i)
            obs = env.reset()
            ep_ret = 0.0
            recent_z = []
            while not env.done:
                x_cur = torch.tensor([obs[0]], dtype=torch.long, device=device)
                z0 = model.world_model.encode_obs(x_cur)
                recent_z.append(z0)
                act = select_action_mpc(model, z0, horizon=horizon, recent_z=recent_z, mode=mode)
                obs, rew, done, is_trap = env.step(act)
                ep_ret += rew
            solved += int(env.agent == env.goal and env.door_open)
            trapped += int(env.trapped)
            returns.append(ep_ret)
        results[mode_name] = {
            "goal_rate": round(solved / n_maps, 4),
            "trap_rate": round(trapped / n_maps, 4),
            "mean_return": round(float(np.mean(returns)), 4),
        }
    return results
