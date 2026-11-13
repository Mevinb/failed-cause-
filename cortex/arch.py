import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CortexConfig:
    vocab_size: int = 256
    max_seq_len: int = 512
    d_model: int = 256
    n_head: int = 4
    local_window: int = 48
    micro_chunk: int = 16
    mem_dim: int = 32
    ponder_steps: int = 4
    max_ponder: int = 12
    grid_max: int = 8
    n_slots: int = 48
    slot_merge_sim: float = 0.92


def build_rope(seq_len: int, head_dim: int, base: float = 10000.0) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(seq_len).float(), inv)
    return torch.stack([ang.cos(), ang.sin()], dim=-1)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    a, b = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = rope[: x.shape[-2]].to(x.dtype).unbind(-1)
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)


def make_causal_mask(seq_len: int, window: int | None) -> torch.Tensor:
    idx = torch.arange(seq_len)
    diff = idx[:, None] - idx[None, :]
    if window is None:
        return diff >= 0
    return (diff >= 0) & (diff < window)


class MultimodalPerceptionLanguage(nn.Module):
    """Subsystems 1, 2, 10: Unified discrete token/symbol interface + 2D axial spatial grid perception."""

    def __init__(self, cfg: CortexConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.row_emb = nn.Embedding(cfg.grid_max + 1, cfg.d_model)
        self.col_emb = nn.Embedding(cfg.grid_max + 1, cfg.d_model)
        self.mod_emb = nn.Embedding(4, cfg.d_model)  # 0=seq/lang, 1=2D-grid, 2=causal-graph, 3=embodied
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.row_emb.weight, std=0.01)
        nn.init.normal_(self.col_emb.weight, std=0.01)
        nn.init.normal_(self.mod_emb.weight, std=0.01)

    def forward(
        self,
        idx: torch.Tensor,
        rows: torch.Tensor | None = None,
        cols: torch.Tensor | None = None,
        modality: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        x = self.tok_emb(idx)
        if isinstance(modality, int):
            x = x + self.mod_emb.weight[modality].view(1, 1, -1)
        else:
            x = x + self.mod_emb(modality)
        if rows is not None and cols is not None:
            x = x + self.row_emb(rows) + self.col_emb(cols)
        return x


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CausalAttention(nn.Module):
    def __init__(self, cfg: CortexConfig):
        super().__init__()
        self.n_head = cfg.n_head
        hd = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.q_norm = nn.RMSNorm(hd)
        self.k_norm = nn.RMSNorm(hd)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, rope: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.q_norm(q), rope), apply_rope(self.k_norm(k), rope)
        if mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            m = mask[:T, :T] if mask.dim() == 2 else mask[:, :, :T, :T]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SurpriseDeltaMemoryAttention(nn.Module):
    """Subsystems 3 & 9: Local QK-Norm RoPE Working Attention + Bounded Surprise-Gated Delta Fast Weights."""

    def __init__(self, cfg: CortexConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.mem_dim = cfg.mem_dim
        self.micro_chunk = cfg.micro_chunk
        H, Dm = cfg.n_head, cfg.mem_dim
        hd = cfg.d_model // H
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.q_norm = nn.RMSNorm(hd)
        self.k_norm = nn.RMSNorm(hd)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.qkv_mem = nn.Linear(cfg.d_model, 3 * H * Dm, bias=False)
        self.mem_proj = nn.Linear(H * Dm, cfg.d_model, bias=False)
        self.gates = nn.Linear(cfg.d_model, 2 * H)
        self.surprise_gain = nn.Parameter(torch.full((H,), 1.5))
        self.decay_logit = nn.Parameter(torch.full((H,), 5.5))
        self.read_logit = nn.Parameter(torch.full((H,), 0.5))
        nn.init.zeros_(self.gates.weight)
        nn.init.constant_(self.gates.bias[:H], -2.2)
        nn.init.zeros_(self.gates.bias[H:])

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        local_mask: torch.Tensor,
        reset_mask: torch.Tensor,
        mem: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        H, Dm, MC = self.n_head, self.mem_dim, self.micro_chunk
        q, k, v = self.qkv(x).view(B, T, 3, H, C // H).permute(2, 0, 3, 1, 4)
        m_loc = local_mask[:T, :T] if attn_mask is None else attn_mask[:, :, :T, :T]
        y_loc = F.scaled_dot_product_attention(apply_rope(self.q_norm(q), rope), apply_rope(self.k_norm(k), rope), v, attn_mask=m_loc)
        y_loc = self.proj(y_loc.transpose(1, 2).reshape(B, T, C))

        qkv_m = self.qkv_mem(x)
        g = self.gates(x)
        with torch.autocast(x.device.type, enabled=False):
            qm, km, vm = qkv_m.float().view(B, T, 3, H, Dm).permute(2, 0, 3, 1, 4)
            qm, km = F.normalize(qm, dim=-1), F.normalize(km, dim=-1)
            vm = F.normalize(vm, dim=-1) * math.sqrt(Dm)
            beta_b, decay_m = g.float().view(B, T, 2, H).permute(2, 0, 3, 1).unsqueeze(-1).unbind(0)
            if mem is None:
                mem = torch.zeros(B, H, Dm, Dm, device=x.device)
            reads, surprises = [], []
            for s in range(0, T, MC):
                qc, kc, vc = qm[:, :, s : s + MC], km[:, :, s : s + MC], vm[:, :, s : s + MC]
                ec = reset_mask[:, s : s + MC]
                same_ep = (ec.cumsum(dim=1) == 0)[:, None, :, None].float()
                reads.append(same_ep * (qc @ mem))

                keep_prev = (~ec.any(dim=1))[:, None, None, None].float()
                rev = ec.flip(1).cumsum(1).flip(1)
                active = ((rev == 0) | (ec & (rev == 1)))[:, None, :, None].float()
                decay = torch.sigmoid(self.decay_logit.view(1, H, 1, 1) + decay_m[:, :, s : s + MC].mean(dim=2, keepdim=True))
                mem_d = decay * (keep_prev * mem)
                err = vc - kc @ mem_d
                surp = err.detach().norm(dim=-1, keepdim=True) / math.sqrt(Dm)
                surprises.append(surp.mean(dim=1))
                beta = active * torch.sigmoid(beta_b[:, :, s : s + MC] + self.surprise_gain.view(1, H, 1, 1) * surp)
                mem = mem_d + (kc.transpose(-1, -2) @ (beta * err)) / (1.0 + beta.sum(dim=2, keepdim=True))
            read = torch.sigmoid(self.read_logit).view(1, H, 1, 1) * torch.cat(reads, dim=2)
            mean_surp = torch.cat(surprises, dim=1).squeeze(-1)
        y_mem = self.mem_proj(read.to(x.dtype).transpose(1, 2).reshape(B, T, H * Dm))
        return y_loc + y_mem, mem, mean_surp


class HippocampalMemoryAttention(SurpriseDeltaMemoryAttention):
    """Gen-2: Surprise-Gated Delta Matrix + Sparse Competitive Landmark Slot Bank.

    The 32x32 matrix forgets smoothly with distance; discrete slots never decay —
    retrieval is a modern-Hopfield softmax match, so a fact written 800 tokens ago is
    as sharp as one written 8 ago. Top-1 surprising token per micro-chunk writes a slot;
    near-duplicate keys merge by EMA; least-salient slot is evicted; slots wipe on reset.
    """

    n_slots = 48
    merge_sim = 0.92

    def __init__(self, cfg: CortexConfig):
        super().__init__(cfg)
        self.n_slots = cfg.n_slots
        self.merge_sim = cfg.slot_merge_sim
        H = cfg.n_head
        self.slot_gate = nn.Linear(cfg.d_model, H)
        self.slot_temp = nn.Parameter(torch.full((H,), 2.0))
        nn.init.zeros_(self.slot_gate.weight)
        nn.init.constant_(self.slot_gate.bias, -0.5)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        local_mask: torch.Tensor,
        reset_mask: torch.Tensor,
        mem: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        H, Dm, MC, S = self.n_head, self.mem_dim, self.micro_chunk, self.n_slots
        q, k, v = self.qkv(x).view(B, T, 3, H, C // H).permute(2, 0, 3, 1, 4)
        m_loc = local_mask[:T, :T] if attn_mask is None else attn_mask[:, :, :T, :T]
        y_loc = F.scaled_dot_product_attention(apply_rope(self.q_norm(q), rope), apply_rope(self.k_norm(k), rope), v, attn_mask=m_loc)
        y_loc = self.proj(y_loc.transpose(1, 2).reshape(B, T, C))

        qkv_m = self.qkv_mem(x)
        g = self.gates(x)
        sg_in = self.slot_gate(x)
        with torch.autocast(x.device.type, enabled=False):
            qm, km, vm = qkv_m.float().view(B, T, 3, H, Dm).permute(2, 0, 3, 1, 4)
            qm, km = F.normalize(qm, dim=-1), F.normalize(km, dim=-1)
            vm = F.normalize(vm, dim=-1) * math.sqrt(Dm)
            beta_b, decay_m = g.float().view(B, T, 2, H).permute(2, 0, 3, 1).unsqueeze(-1).unbind(0)
            sg = torch.sigmoid(sg_in.float()).transpose(1, 2).unsqueeze(-1)
            temp = F.softplus(self.slot_temp).view(1, H, 1, 1)
            if state is None:
                state = {
                    "M": torch.zeros(B, H, Dm, Dm, device=x.device),
                    "K": torch.zeros(B, H, S, Dm, device=x.device),
                    "V": torch.zeros(B, H, S, Dm, device=x.device),
                    "w": torch.zeros(B, H, S, device=x.device),
                }
            mem, sk, sv, sw = state["M"], state["K"], state["V"], state["w"]
            reads, surprises = [], []
            for s in range(0, T, MC):
                qc, kc, vc = qm[:, :, s : s + MC], km[:, :, s : s + MC], vm[:, :, s : s + MC]
                ec = reset_mask[:, s : s + MC]
                same_ep = (ec.cumsum(dim=1) == 0)[:, None, :, None].float()

                slot_score = temp * (qc @ sk.transpose(-1, -2))
                slot_score = slot_score.masked_fill((sw <= 1e-6)[:, :, None, :], -1e4)
                r_slot = torch.softmax(slot_score, dim=-1) @ sv
                reads.append(same_ep * (qc @ mem + sg[:, :, s : s + MC] * r_slot))

                keep_prev = (~ec.any(dim=1))[:, None, None, None].float()
                rev = ec.flip(1).cumsum(1).flip(1)
                active = (rev == 0) | (ec & (rev == 1))
                active_f = active[:, None, :, None].float()
                decay = torch.sigmoid(self.decay_logit.view(1, H, 1, 1) + decay_m[:, :, s : s + MC].mean(dim=2, keepdim=True))
                mem_d = decay * (keep_prev * mem)
                err = vc - kc @ mem_d
                surp = err.detach().norm(dim=-1, keepdim=True) / math.sqrt(Dm)
                surprises.append(surp.mean(dim=1))
                beta = active_f * torch.sigmoid(beta_b[:, :, s : s + MC] + self.surprise_gain.view(1, H, 1, 1) * surp)
                mem = mem_d + (kc.transpose(-1, -2) @ (beta * err)) / (1.0 + beta.sum(dim=2, keepdim=True))

                dead = ec.any(dim=1)[:, None, None, None].float()
                sk, sv = sk * (1 - dead), sv * (1 - dead)
                sw = sw * (1 - dead.squeeze(-1))
                surp_m = surp.squeeze(-1).masked_fill(~active[:, None, :].expand(-1, H, -1), -1e9)
                star = surp_m.argmax(dim=2)
                ix = star[:, :, None, None].expand(-1, -1, 1, Dm)
                k_star, v_star = kc.gather(2, ix).squeeze(2), vc.gather(2, ix).squeeze(2)
                b_star = beta.squeeze(-1).gather(2, star.unsqueeze(-1)).squeeze(-1)
                sim = (k_star.unsqueeze(2) * sk).sum(-1).masked_fill(sw <= 1e-6, -1e9)
                best_sim, best_ix = sim.max(dim=-1)
                do_merge = (best_sim > self.merge_sim) & (sw.gather(-1, best_ix.unsqueeze(-1)).squeeze(-1) > 1e-6)
                has_empty = sw <= 1e-6
                target = torch.where(do_merge, best_ix, torch.where(has_empty.any(dim=-1), has_empty.float().argmax(dim=-1), sw.argmin(dim=-1)))
                t_one = F.one_hot(target, S).float().unsqueeze(-1)
                mrg = do_merge[:, :, None, None]
                new_k = torch.where(mrg, F.normalize(0.7 * sk + 0.3 * k_star.unsqueeze(2), dim=-1), k_star.unsqueeze(2).expand(-1, -1, S, -1))
                new_v = torch.where(mrg, 0.7 * sv + 0.3 * v_star.unsqueeze(2), v_star.unsqueeze(2).expand(-1, -1, S, -1))
                sk = sk * (1 - t_one) + new_k * t_one
                sv = sv * (1 - t_one) + new_v * t_one
                new_w = torch.where(do_merge.unsqueeze(-1), torch.maximum(sw, b_star.unsqueeze(-1)), b_star.unsqueeze(-1).expand(-1, -1, S))
                sw = sw * (1 - t_one.squeeze(-1)) + new_w * t_one.squeeze(-1)
            read = torch.sigmoid(self.read_logit).view(1, H, 1, 1) * torch.cat(reads, dim=2)
            mean_surp = torch.cat(surprises, dim=1).squeeze(-1)
            state = {"M": mem, "K": sk, "V": sv, "w": sw}
        y_mem = self.mem_proj(read.to(x.dtype).transpose(1, 2).reshape(B, T, H * Dm))
        return y_loc + y_mem, mem, mean_surp


class StandardBlock(nn.Module):
    def __init__(self, cfg: CortexConfig, mlp_hidden: int):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.attn = CausalAttention(cfg)
        self.mlp = SwiGLU(cfg.d_model, mlp_hidden)

    def forward(self, x: torch.Tensor, rope: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, mask)
        return x + self.mlp(self.norm2(x))


class DualMemoryBlock(nn.Module):
    def __init__(self, cfg: CortexConfig, mlp_hidden: int):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.attn = SurpriseDeltaMemoryAttention(cfg)
        self.mlp = SwiGLU(cfg.d_model, mlp_hidden)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        local_mask: torch.Tensor,
        reset_mask: torch.Tensor,
        mem: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, mem, surp = self.attn(self.norm1(x), rope, local_mask, reset_mask, mem, attn_mask=attn_mask)
        x = x + a
        return x + self.mlp(self.norm2(x)), mem, surp


class DualMemoryBlockV2(DualMemoryBlock):
    def __init__(self, cfg: CortexConfig, mlp_hidden: int):
        super().__init__(cfg, mlp_hidden)
        self.attn = HippocampalMemoryAttention(cfg)


class IterativeLatentReasoner(nn.Module):
    """Subsystems 5, 8, 11: Input-Anchored Recursive Workspace + Direct Residual Refinement + Epistemic Verifier."""

    def __init__(self, cfg: CortexConfig, mlp_hidden: int):
        super().__init__()
        self.cfg = cfg
        self.inj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.step_emb = nn.Embedding(cfg.max_ponder, cfg.d_model)
        self.core_a = StandardBlock(cfg, mlp_hidden)
        self.core_b = StandardBlock(cfg, mlp_hidden)
        self.step_gate = nn.Parameter(torch.full((cfg.max_ponder,), 1.0))
        self.verifier = nn.Sequential(
            nn.Linear(3 * cfg.d_model + 1, cfg.d_model // 2),
            nn.SiLU(),
            nn.Linear(cfg.d_model // 2, 1),
        )
        nn.init.normal_(self.step_emb.weight, std=0.02)

    def forward(
        self,
        h0: torch.Tensor,
        rope: torch.Tensor,
        steps: int | None = None,
        attn_mask: torch.Tensor | None = None,
        clamp_mask: torch.Tensor | None = None,
        clamp_val: torch.Tensor | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        K = self.cfg.ponder_steps if steps is None else steps
        z = h0
        ys, confs, dzs = [], [], []
        for k in range(K):
            idx_k = min(k, self.cfg.max_ponder - 1)
            delta_h = z - h0
            u = z + self.inj(torch.cat([h0, delta_h], dim=-1)) + self.step_emb.weight[idx_k].view(1, 1, -1)
            z_cand = self.core_b(self.core_a(u, rope, attn_mask), rope, attn_mask)
            gate = torch.sigmoid(self.step_gate[idx_k])
            z_next = z + gate * (z_cand - u)
            if clamp_mask is not None and clamp_val is not None:
                z_next = torch.where(clamp_mask.unsqueeze(-1), clamp_val, z_next)
            dz = (z_next - z).norm(dim=-1, keepdim=True) / math.sqrt(self.cfg.d_model)
            z = z_next
            ver_in = torch.cat([h0.detach(), z.detach(), (z - h0).detach(), dz.detach()], dim=-1)
            conf = self.verifier(ver_in).squeeze(-1)
            ys.append(z)
            confs.append(conf)
            dzs.append(dz.squeeze(-1))
        return {"states": ys, "conf_logits": confs, "dz": dzs, "final_z": z}


def sinusoidal_step_encoding(k: int, d_model: int, device, base: float = 10000.0) -> torch.Tensor:
    """Continuous depth encoding: defined for ANY K, so unrolling past the training
    depth stays on-manifold instead of hitting an untrained embedding row."""
    inv = 1.0 / (base ** (torch.arange(0, d_model, 2, device=device).float() / d_model))
    ang = k * inv
    e = torch.zeros(d_model, device=device)
    e[0::2], e[1::2] = torch.sin(ang), torch.cos(ang)
    return e


class SinusoidalLatentReasoner(IterativeLatentReasoner):
    """Gen-2 Wall-3 fix: continuous depth conditioning + contractive gating + FiLM
    task-program conditioning. Discrete step_emb[k>max_ponder] was untrained noise;
    sinusoidal depth generalizes to K=16+. Optional `program` vector FiLM-modulates
    each refinement step — the substrate for test-time program induction."""

    def __init__(self, cfg: CortexConfig, mlp_hidden: int):
        super().__init__(cfg, mlp_hidden)
        del self.step_emb
        self.step_gain = nn.Parameter(torch.tensor(0.5))
        self.film = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=True)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(
        self,
        h0: torch.Tensor,
        rope: torch.Tensor,
        steps: int | None = None,
        attn_mask: torch.Tensor | None = None,
        clamp_mask: torch.Tensor | None = None,
        clamp_val: torch.Tensor | None = None,
        program: torch.Tensor | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        K = self.cfg.ponder_steps if steps is None else steps
        z = h0
        ys, confs, dzs = [], [], []
        if program is None:
            film_gb = None
        else:
            film_gb = self.film(program)
            if film_gb.dim() == 1:
                film_gb = film_gb[None, None, :]
            elif film_gb.dim() == 2:
                film_gb = film_gb[:, None, :]
        for k in range(K):
            idx_k = min(k, self.cfg.max_ponder - 1)
            delta_h = z - h0
            u = z + self.inj(torch.cat([h0, delta_h], dim=-1))
            u = u + self.step_gain * sinusoidal_step_encoding(k, self.cfg.d_model, u.device)
            if film_gb is not None:
                gam, bet = film_gb.chunk(2, dim=-1)
                u = u * (1.0 + 0.1 * torch.tanh(gam)) + 0.1 * bet
            z_cand = self.core_b(self.core_a(u, rope, attn_mask), rope, attn_mask)
            gate = torch.sigmoid(self.step_gate[idx_k])
            z_next = z + gate * (z_cand - u)
            if clamp_mask is not None and clamp_val is not None:
                z_next = torch.where(clamp_mask.unsqueeze(-1), clamp_val, z_next)
            dz = (z_next - z).norm(dim=-1, keepdim=True) / math.sqrt(self.cfg.d_model)
            z = z_next
            ver_in = torch.cat([h0.detach(), z.detach(), (z - h0).detach(), dz.detach()], dim=-1)
            conf = self.verifier(ver_in).squeeze(-1)
            ys.append(z)
            confs.append(conf)
            dzs.append(dz.squeeze(-1))
        return {"states": ys, "conf_logits": confs, "dz": dzs, "final_z": z}


class LatentWorldModel(nn.Module):
    """Subsystems 4 & 6: Grounded Spatial-State JEPA + Observation Decoder + Reward/Trap/Value/Policy + Subgoal Heads."""

    def __init__(self, d_model: int, n_actions: int = 8, obs_len: int = 39, vocab_size: int = 256):
        super().__init__()
        self.obs_len = obs_len
        self.n_cell_types = 8
        self.obs_tok = nn.Embedding(vocab_size, 48)
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_len * 48, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.obs_dec = nn.Linear(d_model, obs_len * self.n_cell_types)
        self.act_emb = nn.Embedding(n_actions, d_model)
        self.norm_state = nn.RMSNorm(d_model)
        self.trans = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.reward_head = nn.Sequential(nn.Linear(2 * d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 1))
        self.trap_head = nn.Sequential(nn.Linear(2 * d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 1))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 1))
        self.policy_head = nn.Linear(d_model, n_actions)
        # Gen-2 Wall-2: subgoal machinery for hierarchical planning.
        # milestone_head(z): current achievements [has_key, door_open, done]
        # progress_head(z,a): P(next transition achieves [key, door, goal])
        self.milestone_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 3))
        self.progress_head = nn.Sequential(nn.Linear(2 * d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 3))

    def encode_obs(self, idx: torch.Tensor) -> torch.Tensor:
        flat = self.obs_tok(idx).flatten(1)
        return self.norm_state(self.obs_enc(flat))

    def encode_pool(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm_state(h.mean(dim=1))

    def step_latent(self, z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        za = torch.cat([z, self.act_emb(action)], dim=-1)
        z_next = self.norm_state(z + self.trans(za))
        rew = self.reward_head(za).squeeze(-1)
        trap_logit = self.trap_head(za).squeeze(-1)
        return z_next, rew, trap_logit


class FastTestTimeAdapter(nn.Module):
    """Subsystem 7 (Fast Plasticity): Low-rank bottleneck adapter updated at test time without touching slow weights."""

    def __init__(self, d_model: int, rank: int = 16):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        self.reset_adapter()

    def reset_adapter(self):
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))


class BaseCognitiveModel(nn.Module):
    def __init__(self, cfg: CortexConfig):
        super().__init__()
        self.cfg = cfg
        self.perc = MultimodalPerceptionLanguage(cfg)
        self.norm_out = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.perc.tok_emb.weight
        self.ttt_adapter = FastTestTimeAdapter(cfg.d_model, rank=16)
        self.world_model = LatentWorldModel(cfg.d_model, n_actions=5)
        self.register_buffer("rope", build_rope(cfg.max_seq_len, cfg.d_model // cfg.n_head), persistent=False)
        self.register_buffer("local_mask", make_causal_mask(cfg.max_seq_len, cfg.local_window), persistent=False)
        self.register_buffer("full_mask", make_causal_mask(cfg.max_seq_len, None), persistent=False)

    def episode_local_mask(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        ep_id = (idx == 0).cumsum(dim=1)
        same_ep = ep_id[:, None, :, None] == ep_id[:, None, None, :]
        return same_ep & self.local_mask[:T, :T]

    def _init_residual(self, scale_depth: int):
        for name, p in self.named_parameters():
            if p.dim() >= 2 and not name.startswith(("perc.", "ttt_adapter.", "world_model.")):
                res = name.endswith(("proj.weight", "mem_proj.weight", "down.weight", "cand_upd.weight"))
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * scale_depth) if res else 0.02)
        self.ttt_adapter.reset_adapter()


class H0DenseTransformer(BaseCognitiveModel):
    """H0 Baseline: Windowed RoPE Transformer (W=48), fixed 6-layer depth, softmax confidence only."""

    def __init__(self, cfg: CortexConfig):
        super().__init__(cfg)
        self.blocks = nn.ModuleList(StandardBlock(cfg, mlp_hidden=688) for _ in range(6))
        self._init_residual(6)

    def forward(
        self,
        idx: torch.Tensor,
        rows: torch.Tensor | None = None,
        cols: torch.Tensor | None = None,
        modality: int = 0,
        ponder_steps: int | None = None,
        **kwargs,
    ) -> dict:
        x = self.perc(idx, rows, cols, modality)
        for b in self.blocks:
            x = b(x, self.rope, self.local_mask)
        x = x + self.ttt_adapter(x)
        h = self.norm_out(x)
        logits = self.lm_head(h)
        probs = F.softmax(logits.detach(), dim=-1)
        conf_logit = torch.logit(probs.max(dim=-1).values.clamp(1e-4, 1 - 1e-4))
        return {"logits": logits, "step_logits": [logits], "conf_logits": [conf_logit], "latent": h}


class H1SynapseMem(BaseCognitiveModel):
    """H1 Architecture: Local working attention (W=48) + Surprise-Gated Episode-Aware Delta Fast-Weight Memory."""

    def __init__(self, cfg: CortexConfig):
        super().__init__(cfg)
        self.blocks = nn.ModuleList(DualMemoryBlock(cfg, mlp_hidden=528) for _ in range(6))
        self._init_residual(6)

    def forward(
        self,
        idx: torch.Tensor,
        rows: torch.Tensor | None = None,
        cols: torch.Tensor | None = None,
        modality: int = 0,
        ponder_steps: int | None = None,
        **kwargs,
    ) -> dict:
        reset_mask = idx == 0
        ep_mask = self.episode_local_mask(idx)
        x = self.perc(idx, rows, cols, modality)
        surps = []
        for b in self.blocks:
            x, _, s = b(x, self.rope, self.local_mask, reset_mask, None, attn_mask=ep_mask)
            surps.append(s)
        x = x + self.ttt_adapter(x)
        h = self.norm_out(x)
        logits = self.lm_head(h)
        probs = F.softmax(logits.detach(), dim=-1)
        conf_logit = torch.logit(probs.max(dim=-1).values.clamp(1e-4, 1 - 1e-4))
        return {"logits": logits, "step_logits": [logits], "conf_logits": [conf_logit], "latent": h, "surprise": surps[-1]}


class H2LatentPonderWM(BaseCognitiveModel):
    """H2 Architecture: Windowed Encoder (W=48) + Input-Anchored Iterative Latent Reasoner + Verifier (no fast-weight mem)."""

    def __init__(self, cfg: CortexConfig):
        super().__init__(cfg)
        self.enc = nn.ModuleList(StandardBlock(cfg, mlp_hidden=768) for _ in range(3))
        self.reasoner = IterativeLatentReasoner(cfg, mlp_hidden=768)
        self._init_residual(3 + 2 * cfg.ponder_steps)

    def forward(
        self,
        idx: torch.Tensor,
        rows: torch.Tensor | None = None,
        cols: torch.Tensor | None = None,
        modality: int = 0,
        ponder_steps: int | None = None,
        **kwargs,
    ) -> dict:
        x = self.perc(idx, rows, cols, modality)
        for b in self.enc:
            x = b(x, self.rope, self.local_mask)
        out = self.reasoner(x, self.rope, steps=ponder_steps, attn_mask=self.local_mask)
        step_logits = [self.lm_head(self.norm_out(s + self.ttt_adapter(s))) for s in out["states"]]
        return {
            "logits": step_logits[-1],
            "step_logits": step_logits,
            "conf_logits": out["conf_logits"],
            "latent": self.norm_out(out["states"][-1]),
        }


class H3CortexAGI(BaseCognitiveModel):
    """H3 Unified Architecture (SYNAPSE-1): Multimodal Perc + Surprise-Delta Episodic Mem + Iterative Reasoner + Verifier + WM."""

    def __init__(self, cfg: CortexConfig):
        super().__init__(cfg)
        self.mem_blocks = nn.ModuleList(DualMemoryBlock(cfg, mlp_hidden=656) for _ in range(3))
        self.reasoner = IterativeLatentReasoner(cfg, mlp_hidden=704)
        self._init_residual(3 + 2 * cfg.ponder_steps)

    def forward(
        self,
        idx: torch.Tensor,
        rows: torch.Tensor | None = None,
        cols: torch.Tensor | None = None,
        modality: int = 0,
        ponder_steps: int | None = None,
        **kwargs,
    ) -> dict:
        reset_mask = idx == 0
        ep_mask = self.episode_local_mask(idx)
        x = self.perc(idx, rows, cols, modality)
        surps = []
        for b in self.mem_blocks:
            x, _, s = b(x, self.rope, self.local_mask, reset_mask, None, attn_mask=ep_mask)
            surps.append(s)
        out = self.reasoner(x, self.rope, steps=ponder_steps, attn_mask=ep_mask)
        step_logits = [self.lm_head(self.norm_out(s + self.ttt_adapter(s))) for s in out["states"]]
        return {
            "logits": step_logits[-1],
            "step_logits": step_logits,
            "conf_logits": out["conf_logits"],
            "latent": self.norm_out(out["states"][-1]),
            "surprise": surps[-1],
        }


class H4CortexV2(H3CortexAGI):
    """Gen-2 unified architecture: H3 + Hippocampal slot memory + sinusoidal
    contractive reasoner + FiLM program conditioning + subgoal world-model heads."""

    def __init__(self, cfg: CortexConfig):
        super().__init__(cfg)
        self.mem_blocks = nn.ModuleList(DualMemoryBlockV2(cfg, mlp_hidden=656) for _ in range(3))
        self.reasoner = SinusoidalLatentReasoner(cfg, mlp_hidden=640)

    def forward(
        self,
        idx: torch.Tensor,
        rows: torch.Tensor | None = None,
        cols: torch.Tensor | None = None,
        modality: int = 0,
        ponder_steps: int | None = None,
        program: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        reset_mask = idx == 0
        ep_mask = self.episode_local_mask(idx)
        x = self.perc(idx, rows, cols, modality)
        surps = []
        for b in self.mem_blocks:
            x, _, s = b(x, self.rope, self.local_mask, reset_mask, None, attn_mask=ep_mask)
            surps.append(s)
        out = self.reasoner(x, self.rope, steps=ponder_steps, attn_mask=ep_mask, program=program)
        step_logits = [self.lm_head(self.norm_out(s + self.ttt_adapter(s))) for s in out["states"]]
        return {
            "logits": step_logits[-1],
            "step_logits": step_logits,
            "conf_logits": out["conf_logits"],
            "latent": self.norm_out(out["states"][-1]),
            "surprise": surps[-1],
        }


class H3SinusoidalAbl(H3CortexAGI):
    """Mechanism-isolation ablation (Exp #015c follow-up): H3's exact delta-matrix
    memory blocks + the SinusoidalLatentReasoner (no slot bank, no FiLM program use).
    If this groks do() like H4, depth-sharing alone explains the causal breakthrough."""


    def __init__(self, cfg: CortexConfig):
        super().__init__(cfg)
        self.reasoner = SinusoidalLatentReasoner(cfg, mlp_hidden=640)


ARCH_REGISTRY = {
    "H0_transformer": H0DenseTransformer,
    "H1_synapse_mem": H1SynapseMem,
    "H2_latent_ponder": H2LatentPonderWM,
    "H3_cortex_agi": H3CortexAGI,
    "H4_cortex_v2": H4CortexV2,
    "H3_sinusoidal_abl": H3SinusoidalAbl,
}
