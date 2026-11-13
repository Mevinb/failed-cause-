"""8-Domain Cognitive & Algorithmic AGI Benchmark Suite (ID + Strict OOD Splits)."""
import random
from itertools import permutations

import numpy as np
import torch

# Token vocabulary layout (vocab_size = 256)
TOK_RESET = 0
TOK_PAD = 1
TOK_SEP = 2
TOK_QUERY = 3
TOK_OBS = 4
TOK_DO = 5
TOK_IN_GRID = 6
TOK_OUT_GRID = 7
TOK_HINT = 8

TASK_EPISODIC = 10
TASK_S5 = 11
TASK_MATH = 12
TASK_ARC = 13
TASK_CAUSAL = 14
TASK_MM = 15

KEY_BASE = 20       # 20..51 (32 entity keys)
VAL_BASE = 60       # 60..91 (32 discrete values / mod-17 ints 0..16 -> 60..76)
COLOR_BASE = 100    # 100..109 (10 grid colors 0..9)

OP_ADD = 110
OP_SUB = 111
OP_MUL = 112
TOK_LP = 113
TOK_RP = 114
TOK_EQ = 115

SWAP_BASE = 120     # 120..129: 10 transpositions in S_5
FILLER_BASE = 180   # 180..219: 40 distractor tokens
DIR_BASE = 220      # 220..223: N, S, E, W

S5_SWAPS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
MOD_P = 17


def apply_grid_op(g: np.ndarray, op: str) -> np.ndarray:
    if op == "rot90":
        return np.rot90(g, 1)
    if op == "rot180":
        return np.rot90(g, 2)
    if op == "rot270":
        return np.rot90(g, 3)
    if op == "flip_h":
        return np.fliplr(g)
    if op == "flip_v":
        return np.flipud(g)
    if op == "transpose":
        return g.T.copy()
    if op == "shift_right":
        return np.roll(g, 1, axis=1)
    if op == "shift_down":
        return np.roll(g, 1, axis=0)
    if op == "color_swap_12":
        out = g.copy()
        out[g == 1] = 2
        out[g == 2] = 1
        return out
    if op == "color_cycle":
        out = g.copy()
        nz = g > 0
        out[nz] = (g[nz] % 5) + 1
        return out
    if op == "invert_mask":
        out = np.zeros_like(g)
        out[g == 0] = 3
        return out
    if op == "gravity_down":
        out = np.zeros_like(g)
        for c in range(g.shape[1]):
            col = g[:, c][g[:, c] > 0]
            if len(col):
                out[-len(col) :, c] = col
        return out
    # Compositional rules (op1+op2)
    if "+" in op:
        a, b = op.split("+")
        return apply_grid_op(apply_grid_op(g, a), b)
    raise ValueError(f"Unknown grid op: {op}")


ARC_TRAIN_OPS = [
    "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose",
    "shift_right", "shift_down", "color_swap_12", "color_cycle", "invert_mask", "gravity_down",
]
ARC_OOD_OPS = [
    "flip_h+color_cycle",
    "rot90+color_swap_12",
    "flip_v+shift_right",
    "transpose+color_cycle",
]


def sample_episodic(rng: random.Random, ood: str | None = None) -> dict:
    """Axis 1: Unbounded Episodic Key-Value Recall across distractor streams."""
    if ood == "long160":
        n_kv, gap = 6, 160
    elif ood == "long320":
        n_kv, gap = 6, 320
    elif ood == "id":
        n_kv, gap = 4, 64
    else:
        n_kv, gap = rng.choice([3, 4, 5]), rng.randint(44, 92)

    keys = rng.sample(range(16), n_kv)
    vals = [rng.randrange(16) for _ in range(n_kv)]
    seq = [TOK_RESET, TASK_EPISODIC]
    for k, v in zip(keys, vals):
        seq.extend([KEY_BASE + k, VAL_BASE + v])
    seq.append(TOK_SEP)
    for _ in range(gap):
        seq.append(FILLER_BASE + rng.randrange(40))
    q_idx = rng.randrange(n_kv)
    seq.extend([TOK_QUERY, KEY_BASE + keys[q_idx], VAL_BASE + vals[q_idx]])
    # Predict only the final answer token
    loss_mask = [0] * (len(seq) - 2) + [1, 0]
    return {"seq": seq, "loss_mask": loss_mask, "modality": 0, "n_targets": 1, "chance": 1.0 / 16}


def sample_s5(rng: random.Random, ood: str | None = None) -> dict:
    """Axis 2: Non-commutative S_5 permutation state tracking (NC^1-complete, outside TC^0)."""
    if ood == "len10":
        length = 10
    elif ood == "len13":
        length = 13
    elif ood == "id":
        length = 6
    else:
        length = rng.randint(0, 6)

    perm = list(rng.choice(list(permutations(range(5)))))
    seq = [TOK_RESET, TASK_S5] + [VAL_BASE + x for x in perm] + [TOK_SEP]
    for _ in range(length):
        s_idx = rng.randrange(10)
        i, j = S5_SWAPS[s_idx]
        perm[i], perm[j] = perm[j], perm[i]
        seq.append(SWAP_BASE + s_idx)
    q_slot = rng.randrange(5)
    seq.extend([TOK_QUERY, KEY_BASE + q_slot, VAL_BASE + perm[q_slot]])
    loss_mask = [0] * (len(seq) - 2) + [1, 0]
    return {"seq": seq, "loss_mask": loss_mask, "modality": 0, "n_targets": 1, "chance": 0.20}


def _build_expr(rng: random.Random, depth: int, env: dict[int, int]) -> tuple[list[int], int]:
    if depth == 0:
        if env and rng.random() < 0.6:
            k = rng.choice(list(env.keys()))
            return [KEY_BASE + k], env[k]
        v = rng.randrange(MOD_P)
        return [VAL_BASE + v], v
    left_toks, lv = _build_expr(rng, depth - 1, env)
    right_toks, rv = _build_expr(rng, 0 if depth == 1 else rng.randint(0, depth - 1), env)
    op = rng.choice([OP_ADD, OP_SUB, OP_MUL])
    if op == OP_ADD:
        res = (lv + rv) % MOD_P
    elif op == OP_SUB:
        res = (lv - rv) % MOD_P
    else:
        res = (lv * rv) % MOD_P
    return [TOK_LP] + left_toks + [op] + right_toks + [TOK_RP], res


def sample_comp_math(rng: random.Random, ood: str | None = None) -> dict:
    """Axis 3: Compositional modular arithmetic (mod 17) with variable bindings."""
    if ood == "depth3":
        depth = 3
    elif ood == "depth4":
        depth = 4
    elif ood == "id":
        depth = 2
    else:
        depth = rng.choice([1, 2])

    env = {}
    seq = [TOK_RESET, TASK_MATH]
    for k in range(2):
        v = rng.randrange(MOD_P)
        env[k] = v
        seq.extend([KEY_BASE + k, TOK_EQ, VAL_BASE + v, TOK_SEP])
    expr_toks, ans = _build_expr(rng, depth, env)
    seq.extend(expr_toks + [TOK_QUERY, VAL_BASE + ans])
    loss_mask = [0] * (len(seq) - 2) + [1, 0]
    return {"seq": seq, "loss_mask": loss_mask, "modality": 0, "n_targets": 1, "chance": 1.0 / MOD_P}


def _encode_grid_pair(inp: np.ndarray, out: np.ndarray | None) -> tuple[list[int], list[int], list[int]]:
    H, W = inp.shape
    toks, rows, cols = [TOK_IN_GRID], [0], [0]
    for r in range(H):
        for c in range(W):
            toks.append(COLOR_BASE + int(inp[r, c]))
            rows.append(r + 1)
            cols.append(c + 1)
    if out is not None:
        toks.append(TOK_OUT_GRID)
        rows.append(0)
        cols.append(0)
        for r in range(H):
            for c in range(W):
                toks.append(COLOR_BASE + int(out[r, c]))
                rows.append(r + 1)
                cols.append(c + 1)
    return toks, rows, cols


def _random_arc_grid(rng: random.Random, size: int = 4) -> np.ndarray:
    g = np.zeros((size, size), dtype=np.int64)
    n_cells = rng.randint(4, 10)
    positions = rng.sample(range(size * size), n_cells)
    for p in positions:
        g[p // size, p % size] = rng.randint(1, 5)
    return g


def sample_arc_grid(rng: random.Random, ood: str | None = None, specific_op: str | None = None) -> dict:
    """Axis 4: Abstract 2D Visual-Grid Rule Induction (3 support demos -> 1 test grid)."""
    if specific_op is not None:
        op = specific_op
    elif ood == "comp_ood":
        op = rng.choice(ARC_OOD_OPS)
    else:
        op = rng.choice(ARC_TRAIN_OPS)

    seq, rows, cols = [TOK_RESET, TASK_ARC], [0, 0], [0, 0]
    demos = []
    for _ in range(3):
        gi = _random_arc_grid(rng, 4)
        go = apply_grid_op(gi, op)
        demos.append((gi, go))
        t, r, c = _encode_grid_pair(gi, go)
        seq.extend(t)
        rows.extend(r)
        cols.extend(c)
    test_in = _random_arc_grid(rng, 4)
    test_out = apply_grid_op(test_in, op)
    t, r, c = _encode_grid_pair(test_in, test_out)
    # Predict the 16 cells of test_out (which follow TOK_OUT_GRID)
    prefix_len = len(seq) + 1 + 16  # up to and including TOK_OUT_GRID (index prefix_len - 1 predicts test_out[0])
    seq.extend(t)
    rows.extend(r)
    cols.extend(c)
    loss_mask = [0] * len(seq)
    for pos in range(prefix_len, prefix_len + 16):
        loss_mask[pos - 1] = 1
    return {
        "seq": seq,
        "rows": rows,
        "cols": cols,
        "loss_mask": loss_mask,
        "modality": 1,
        "n_targets": 16,
        "chance": (1.0 / 6) ** 16,
        "demos": demos,
        "op": op,
    }


# Confounded SCM parameterizations (base_z0, base_z1, delta_x1) with base_z0 != base_z1 (48 distinct SCMs; 12 held out strictly for OOD)
ALL_CAUSAL_PARAMS = [(b0, b1, dx) for b0 in range(4) for b1 in range(4) if b0 != b1 for dx in range(4)]
CAUSAL_OOD_SET = {p for i, p in enumerate(ALL_CAUSAL_PARAMS) if i % 4 == 3}
CAUSAL_TRAIN_LIST = [p for p in ALL_CAUSAL_PARAMS if p not in CAUSAL_OOD_SET]
CAUSAL_OOD_LIST = [p for p in ALL_CAUSAL_PARAMS if p in CAUSAL_OOD_SET]


def sample_causal(rng: random.Random, ood: str | None = None) -> dict:
    """Axis 5: Confounded Causal Inference — P(Y | X=x) vs P(Y | Z=z0, do(X=x)).

    Observationally Z=0 -> X=0 and Z=1 -> X=1 with 100% confounding.
    Intervening do(X=1) when Z=0 severs Z->X, yielding (base_z0 + delta_x1) mod 4 != Y_obs(X=1).
    """
    b0, b1, dx = rng.choice(CAUSAL_OOD_LIST if ood == "ood_scm" else CAUSAL_TRAIN_LIST)
    y_obs0 = b0
    y_obs1 = (b1 + dx) % 4
    y_do_z1_x0 = b1
    y_do_z0_x1 = (b0 + dx) % 4

    records = [
        [TOK_OBS, VAL_BASE + 0, VAL_BASE + 0, VAL_BASE + y_obs0, TOK_SEP],
        [TOK_OBS, VAL_BASE + 1, VAL_BASE + 1, VAL_BASE + y_obs1, TOK_SEP],
        [TOK_DO, VAL_BASE + 1, VAL_BASE + 0, VAL_BASE + y_do_z1_x0, TOK_SEP],
    ]
    rng.shuffle(records)
    seq = [TOK_RESET, TASK_CAUSAL]
    for rec in records:
        seq.extend(rec)

    is_do = (ood == "ood_do_only") or (rng.random() < 0.5)
    if is_do:
        # Interventional query: Z=0, do(X=1) -> must compose b0 + (y_obs1 - b1) mod 4!
        seq.extend([TOK_DO, VAL_BASE + 0, VAL_BASE + 1, TOK_QUERY, VAL_BASE + y_do_z0_x1])
    else:
        # Observational query: obs(X=x) -> under confounding Z=x
        qx = rng.randrange(2)
        qy = y_obs1 if qx == 1 else y_obs0
        seq.extend([TOK_OBS, VAL_BASE + qx, VAL_BASE + qx, TOK_QUERY, VAL_BASE + qy])
    loss_mask = [0] * (len(seq) - 2) + [1, 0]
    return {"seq": seq, "loss_mask": loss_mask, "modality": 2, "n_targets": 1, "chance": 0.25}


DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # W, E, N, S in (dr, dc) -> let's index 0=N(-1,0),1=S(1,0),2=E(0,1),3=W(0,-1)
DIR_DELTAS = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def sample_multimodal(rng: random.Random, ood: str | None = None) -> dict:
    """Axis 6: Multimodal 2D Grid + Relational Language Grounding (1-hop train -> 2-hop OOD)."""
    hops = 2 if ood == "hops2" else (1 if ood == "id" else rng.choice([1, 1, 1, 2]) if ood == "mixed" else 1)
    size = 5
    g = np.zeros((size, size), dtype=np.int64)
    for r in range(size):
        for c in range(size):
            g[r, c] = rng.randint(1, 6)
    # Place unique landmark color 9 at valid interior cell
    lr = rng.randint(hops, size - 1 - hops)
    lc = rng.randint(hops, size - 1 - hops)
    g[lr, lc] = 9

    dirs = [rng.randrange(4) for _ in range(hops)]
    tr, tc = lr, lc
    for d in dirs:
        dr, dc = DIR_DELTAS[d]
        tr, tc = tr + dr, tc + dc
    # Avoid landing back on landmark in 2-hop opposite steps
    if (tr, tc) == (lr, lc):
        d0 = dirs[0]
        dirs[1] = (d0 + 2) % 4
        tr = lr + DIR_DELTAS[d0][0] + DIR_DELTAS[dirs[1]][0]
        tc = lc + DIR_DELTAS[d0][1] + DIR_DELTAS[dirs[1]][1]
    ans_color = int(g[tr, tc])

    seq, rows, cols = [TOK_RESET, TASK_MM, TOK_IN_GRID], [0, 0, 0], [0, 0, 0]
    for r in range(size):
        for c in range(size):
            seq.append(COLOR_BASE + int(g[r, c]))
            rows.append(r + 1)
            cols.append(c + 1)
    seq.append(TOK_HINT)
    rows.append(0)
    cols.append(0)
    for d in dirs:
        seq.append(DIR_BASE + d)
        rows.append(0)
        cols.append(0)
    seq.extend([TOK_QUERY, COLOR_BASE + ans_color])
    rows.extend([0, 0])
    cols.extend([0, 0])
    loss_mask = [0] * (len(seq) - 2) + [1, 0]
    return {"seq": seq, "rows": rows, "cols": cols, "loss_mask": loss_mask, "modality": 1, "n_targets": 1, "chance": 1.0 / 6}


TRAIN_SAMPLERS = [
    ("episodic", sample_episodic),
    ("s5", sample_s5),
    ("math", sample_comp_math),
    ("arc", sample_arc_grid),
    ("causal", sample_causal),
    ("multimodal", sample_multimodal),
]


def collate_episodes(episodes: list[dict], device: str = "cuda") -> dict[str, torch.Tensor]:
    B = len(episodes)
    max_len = max(len(e["seq"]) - 1 for e in episodes)
    x = torch.full((B, max_len), TOK_PAD, dtype=torch.long)
    y = torch.full((B, max_len), -100, dtype=torch.long)
    rows = torch.zeros((B, max_len), dtype=torch.long)
    cols = torch.zeros((B, max_len), dtype=torch.long)
    mods = torch.zeros((B, 1), dtype=torch.long)

    for i, e in enumerate(episodes):
        L = len(e["seq"]) - 1
        x[i, :L] = torch.tensor(e["seq"][:-1], dtype=torch.long)
        tgt = torch.tensor(e["seq"][1:], dtype=torch.long)
        mask = torch.tensor(e["loss_mask"][:-1], dtype=torch.bool)
        y[i, :L] = torch.where(mask, tgt, torch.full_like(tgt, -100))
        if "rows" in e:
            rows[i, :L] = torch.tensor(e["rows"][:-1], dtype=torch.long)
            cols[i, :L] = torch.tensor(e["cols"][:-1], dtype=torch.long)
        mods[i, 0] = e["modality"]

    return {
        "x": x.to(device),
        "y": y.to(device),
        "rows": rows.to(device),
        "cols": cols.to(device),
        "modality": mods.to(device),
    }
