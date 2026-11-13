import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 6
    d_model: int = 384


def rope_table(seq_len, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(seq_len).float(), inv)
    return torch.stack([ang.cos(), ang.sin()], dim=-1)


def apply_rope(x, rope):
    a, b = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = rope[: x.shape[-2]].to(x.dtype).unbind(-1)
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, rope):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(apply_rope(q, rope), apply_rope(k, rope), v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = 8 * cfg.d_model // 3
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    attn_cls = Attention

    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.attn, self.mlp = self.attn_cls(cfg), MLP(cfg)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    """Baseline decoder-only transformer: pre-norm RMSNorm, RoPE, SwiGLU, tied embeddings."""

    block_cls = Block

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.max_context = cfg.block_size
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(self.block_cls(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight
        self.register_buffer("rope", rope_table(cfg.block_size, cfg.d_model // cfg.n_head), persistent=False)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                residual_out = name.endswith(("proj.weight", "down.weight"))
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer) if residual_out else 0.02)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x, self.rope)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=40, stop=None, generator=None):
        for _ in range(n):
            logits = self(idx if self.max_context is None else idx[:, -self.max_context:])[0][:, -1].float()
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                logits[logits < torch.topk(logits, top_k).values[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            if stop is not None and nxt.item() == stop:
                break
            idx = torch.cat([idx, nxt], dim=1)
        return idx


class MemAttention(Attention):
    """Causal attention inside the window, plus a per-head associative memory carried between windows.

    The memory M (head_dim x head_dim per head) is read by every query of the window and written once at the end of
    the window: each token proposes a write with its own learned strength beta, and M takes one normalised delta-rule
    step towards mapping those keys to their values, after a learned per-head decay.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.write = nn.Linear(cfg.d_model, cfg.n_head)
        self.decay_logit = nn.Parameter(torch.full((cfg.n_head,), 3.0))
        self.read_gate = nn.Parameter(torch.zeros(cfg.n_head))

    def forward(self, x, rope, mem):
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.qkv(x).view(B, T, 3, H, C // H).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(apply_rope(q, rope), apply_rope(k, rope), v, is_causal=True)

        beta_logit = self.write(x)
        # the memory accumulates over many windows, so keep its math in fp32 even under bf16 autocast
        with torch.autocast(x.device.type, enabled=False):
            qn, kn, vf = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1), v.float()
            if mem is None:
                mem = torch.zeros(B, H, C // H, C // H, device=x.device)
            read = self.read_gate.view(1, H, 1, 1) * (qn @ mem)
            beta = torch.sigmoid(beta_logit.float()).transpose(1, 2).unsqueeze(-1)
            update = kn.transpose(-1, -2) @ (beta * (vf - kn @ mem))
            decay = torch.sigmoid(self.decay_logit).view(1, H, 1, 1)
            mem = decay * mem + update / (1 + beta.sum(dim=2, keepdim=True))
        y = y + read.to(y.dtype)
        return self.proj(y.transpose(1, 2).reshape(B, T, C)), mem


class MemBlock(Block):
    attn_cls = MemAttention

    def forward(self, x, rope, mem):
        a, mem = self.attn(self.norm1(x), rope, mem)
        x = x + a
        return x + self.mlp(self.norm2(x)), mem


class MemoryTransformer(Transformer):
    """Baseline transformer whose attention layers also read a memory written by the previous windows.

    Input of any length is processed in windows of block_size, carrying the memory forward, so there is no context
    limit. Training sees `segments` consecutive windows per sequence with gradients flowing through the memory.
    """

    block_cls = MemBlock
    segments = 4

    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_context = None
        for b in self.blocks:
            nn.init.constant_(b.attn.write.bias, -1.0)

    def forward_stream(self, idx, targets=None, mems=None):
        x = self.emb(idx)
        new = []
        for i, block in enumerate(self.blocks):
            x, m = block(x, self.rope, None if mems is None else mems[i])
            new.append(m)
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        return logits, loss, new

    def forward(self, idx, targets=None):
        T, mems, logits, total = self.cfg.block_size, None, [], 0.0
        for s in range(0, idx.shape[1], T):
            lg, loss, mems = self.forward_stream(idx[:, s:s + T], None if targets is None else targets[:, s:s + T], mems)
            logits.append(lg)
            if targets is not None:
                total = total + loss * lg.shape[1]
        logits = torch.cat(logits, dim=1)
        return logits, None if targets is None else total / idx.shape[1]


class SynapseAttention(nn.Module):
    """Local sliding-window RoPE working attention + decoupled, episode-aware, surprise-gated Delta fast-weight memory."""

    micro_chunk = 32
    mem_head_dim = 32

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        H, Dm = cfg.n_head, self.mem_head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.qkv_mem = nn.Linear(cfg.d_model, 3 * H * Dm, bias=False)
        self.mem_proj = nn.Linear(H * Dm, cfg.d_model, bias=False)
        self.mem_gates = nn.Linear(cfg.d_model, 2 * H)
        self.surprise_gain = nn.Parameter(torch.full((H,), 1.5))
        self.decay_logit = nn.Parameter(torch.full((H,), 5.5))
        self.read_logit = nn.Parameter(torch.full((H,), 0.5))

    def forward(self, x, rope, local_mask, eot, mem):
        B, T, C = x.shape
        H, Dm, MC = self.n_head, self.mem_head_dim, self.micro_chunk
        q, k, v = self.qkv(x).view(B, T, 3, H, C // H).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(apply_rope(q, rope), apply_rope(k, rope), v, attn_mask=local_mask[:T, :T])
        y = self.proj(y.transpose(1, 2).reshape(B, T, C))

        qkv_m = self.qkv_mem(x)
        gates = self.mem_gates(x)
        with torch.autocast(x.device.type, enabled=False):
            qm, km, vm = qkv_m.float().view(B, T, 3, H, Dm).permute(2, 0, 3, 1, 4)
            qm, km = F.normalize(qm, dim=-1), F.normalize(km, dim=-1)
            vm = F.normalize(vm, dim=-1) * math.sqrt(Dm)
            beta_b, decay_m = gates.float().view(B, T, 2, H).permute(2, 0, 3, 1).unsqueeze(-1).unbind(0)
            if mem is None:
                mem = torch.zeros(B, H, Dm, Dm, device=x.device)
            reads = []
            for s in range(0, T, MC):
                qc, kc, vc = qm[:, :, s:s + MC], km[:, :, s:s + MC], vm[:, :, s:s + MC]
                ec = eot[:, s:s + MC]
                same_ep = (ec.cumsum(dim=1) == 0)[:, None, :, None].float()
                reads.append(same_ep * (qc @ mem))

                keep_prev = (~ec.any(dim=1))[:, None, None, None].float()
                rev = ec.flip(1).cumsum(1).flip(1)
                active = ((rev == 0) | (ec & (rev == 1)))[:, None, :, None].float()
                decay = torch.sigmoid(self.decay_logit.view(1, H, 1, 1) + decay_m[:, :, s:s + MC].mean(dim=2, keepdim=True))
                mem_d = decay * (keep_prev * mem)
                err = vc - kc @ mem_d
                surp = err.detach().norm(dim=-1, keepdim=True) / math.sqrt(Dm)
                beta = active * torch.sigmoid(beta_b[:, :, s:s + MC] + self.surprise_gain.view(1, H, 1, 1) * surp)
                mem = mem_d + (kc.transpose(-1, -2) @ (beta * err)) / (1.0 + beta.sum(dim=2, keepdim=True))
            read = torch.sigmoid(self.read_logit).view(1, H, 1, 1) * torch.cat(reads, dim=2)
        read_out = self.mem_proj(read.to(x.dtype).transpose(1, 2).reshape(B, T, H * Dm))
        return y + read_out, mem


class SynapseMLP(nn.Module):
    def __init__(self, cfg, hidden):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SynapseBlock(nn.Module):
    def __init__(self, cfg, mlp_hidden=928):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.attn, self.mlp = SynapseAttention(cfg), SynapseMLP(cfg, mlp_hidden)

    def forward(self, x, rope, local_mask, eot, mem):
        a, mem = self.attn(self.norm1(x), rope, local_mask, eot, mem)
        x = x + a
        return x + self.mlp(self.norm2(x)), mem


def local_causal_mask(seq_len, window):
    idx = torch.arange(seq_len)
    diff = idx[:, None] - idx[None, :]
    return (diff >= 0) & (diff < window)


class SynapseTransformer(nn.Module):
    """H1 Architecture: Local working attention (W=64) + decoupled episode-aware surprise-gated Delta memory."""

    segments = 4
    local_window = 64

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.max_context = None
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(SynapseBlock(cfg, mlp_hidden=768) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight
        self.register_buffer("rope", rope_table(cfg.block_size, cfg.d_model // cfg.n_head), persistent=False)
        self.register_buffer("local_mask", local_causal_mask(cfg.block_size, self.local_window), persistent=False)
        self._init_weights(cfg.n_layer)

    def _init_weights(self, depth_scale):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                residual_out = name.endswith(("proj.weight", "mem_proj.weight", "down.weight"))
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * depth_scale) if residual_out else 0.02)
        for b in self.blocks:
            nn.init.zeros_(b.attn.mem_gates.weight)
            nn.init.constant_(b.attn.mem_gates.bias[: self.cfg.n_head], -2.2)
            nn.init.zeros_(b.attn.mem_gates.bias[self.cfg.n_head :])

    def forward_stream(self, idx, targets=None, mems=None):
        eot = idx == 0
        x = self.emb(idx)
        new = []
        for i, block in enumerate(self.blocks):
            x, m = block(x, self.rope, self.local_mask, eot, None if mems is None else mems[i])
            new.append(m)
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        return logits, loss, new

    def forward(self, idx, targets=None):
        T, mems, logits, total = self.cfg.block_size, None, [], 0.0
        for s in range(0, idx.shape[1], T):
            lg, loss, mems = self.forward_stream(idx[:, s:s + T], None if targets is None else targets[:, s:s + T], mems)
            logits.append(lg)
            if targets is not None:
                total = total + loss * lg.shape[1]
        logits = torch.cat(logits, dim=1)
        return logits, None if targets is None else total / idx.shape[1]

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=40, stop=None, generator=None):
        for _ in range(n):
            logits = self(idx)[0][:, -1].float()
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                logits[logits < torch.topk(logits, top_k).values[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            if stop is not None and nxt.item() == stop:
                break
            idx = torch.cat([idx, nxt], dim=1)
        return idx


class HippocampusAttention(SynapseAttention):
    """Synapse attention + sparse competitive slot memory (Gen-2 Wall-1 fix).

    Per micro-chunk, the single most surprising token writes its (key, value) into a
    content-addressable slot bank. Slots do not decay with distance: retrieval is a
    modern-Hopfield softmax match, so a fact stored 800 tokens back is as sharp as one
    stored 8 tokens back. Eviction replaces the least-salient slot; near-duplicate
    keys merge by EMA. Slots wipe on episode boundary exactly like the matrix memory.
    """

    n_slots = 48
    merge_sim = 0.92
    n_write_per_chunk = 1  # Gen-2 lever: >1 writes the top-k surprising tokens per chunk

    def __init__(self, cfg):
        super().__init__(cfg)
        H = cfg.n_head
        self.slot_gate = nn.Linear(cfg.d_model, H)
        self.slot_temp = nn.Parameter(torch.full((H,), 2.0))
        nn.init.zeros_(self.slot_gate.weight)
        nn.init.constant_(self.slot_gate.bias, -0.5)

    def forward(self, x, rope, local_mask, eot, state):
        B, T, C = x.shape
        H, Dm, MC, S = self.n_head, self.mem_head_dim, self.micro_chunk, self.n_slots
        q, k, v = self.qkv(x).view(B, T, 3, H, C // H).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(apply_rope(q, rope), apply_rope(k, rope), v, attn_mask=local_mask[:T, :T])
        y = self.proj(y.transpose(1, 2).reshape(B, T, C))

        qkv_m = self.qkv_mem(x)
        gates = self.mem_gates(x)
        sgate = self.slot_gate(x)
        with torch.autocast(x.device.type, enabled=False):
            qm, km, vm = qkv_m.float().view(B, T, 3, H, Dm).permute(2, 0, 3, 1, 4)
            qm, km = F.normalize(qm, dim=-1), F.normalize(km, dim=-1)
            vm = F.normalize(vm, dim=-1) * math.sqrt(Dm)
            beta_b, decay_m = gates.float().view(B, T, 2, H).permute(2, 0, 3, 1).unsqueeze(-1).unbind(0)
            sg = torch.sigmoid(sgate.float()).transpose(1, 2).unsqueeze(-1)
            temp = F.softplus(self.slot_temp).view(1, H, 1, 1)
            if state is None:
                state = {
                    "M": torch.zeros(B, H, Dm, Dm, device=x.device),
                    "K": torch.zeros(B, H, S, Dm, device=x.device),
                    "V": torch.zeros(B, H, S, Dm, device=x.device),
                    "w": torch.zeros(B, H, S, device=x.device),
                }
            mem, sk, sv, sw = state["M"], state["K"], state["V"], state["w"]
            reads = []
            for s in range(0, T, MC):
                qc, kc, vc = qm[:, :, s:s + MC], km[:, :, s:s + MC], vm[:, :, s:s + MC]
                ec = eot[:, s:s + MC]
                same_ep = (ec.cumsum(dim=1) == 0)[:, None, :, None].float()
                L = qc.shape[2]

                # --- dual read: decaying matrix + permanent slot bank ---
                slot_score = temp * (qc @ sk.transpose(-1, -2))
                slot_score = slot_score.masked_fill((sw <= 1e-6)[:, :, None, :], -1e4)
                slot_attn = torch.softmax(slot_score, dim=-1)
                r_slot = slot_attn @ sv
                r_mat = qc @ mem
                reads.append(same_ep * (r_mat + sg[:, :, s:s + MC] * r_slot))

                # --- matrix update (unchanged surprise-gated delta rule) ---
                keep_prev = (~ec.any(dim=1))[:, None, None, None].float()
                rev = ec.flip(1).cumsum(1).flip(1)
                active = ((rev == 0) | (ec & (rev == 1)))
                active_f = active[:, None, :, None].float()
                decay = torch.sigmoid(self.decay_logit.view(1, H, 1, 1) + decay_m[:, :, s:s + MC].mean(dim=2, keepdim=True))
                mem_d = decay * (keep_prev * mem)
                err = vc - kc @ mem_d
                surp = err.detach().norm(dim=-1, keepdim=True) / math.sqrt(Dm)
                beta = active_f * torch.sigmoid(beta_b[:, :, s:s + MC] + self.surprise_gain.view(1, H, 1, 1) * surp)
                mem = mem_d + (kc.transpose(-1, -2) @ (beta * err)) / (1.0 + beta.sum(dim=2, keepdim=True))

                # --- slot update: wipe on EOT, write top-1 surprising active token ---
                dead = ec.any(dim=1)[:, None, None, None].float()  # [B,1,1,1], 0 keeps graph unbroken
                sk, sv = sk * (1 - dead), sv * (1 - dead)
                sw = sw * (1 - dead.squeeze(-1))
                surp_m = surp.squeeze(-1).masked_fill(~active[:, None, :].expand(-1, H, -1), -1e9)
                kw = min(self.n_write_per_chunk, L)
                stars = surp_m.topk(kw, dim=2).indices  # [B,H,kw]
                act_exp = active[:, None, :].expand(-1, H, -1)
                for j in range(kw):
                    star = stars[:, :, j]  # [B,H]
                    ix = star[:, :, None, None].expand(-1, -1, 1, Dm)
                    k_star = kc.gather(2, ix).squeeze(2)
                    v_star = vc.gather(2, ix).squeeze(2)
                    b_star = beta.squeeze(-1).gather(2, star.unsqueeze(-1)).squeeze(-1)
                    # padded picks (inactive positions when fewer than kw actives) must be
                    # full no-ops — otherwise they'd evict a real slot when the bank is full
                    write_ok = act_exp.gather(2, star.unsqueeze(-1)).squeeze(-1).float()  # [B,H]
                    sim = (k_star.unsqueeze(2) * sk).sum(-1)
                    sim = sim.masked_fill(sw <= 1e-6, -1e9)
                    best_sim, best_ix = sim.max(dim=-1)
                    do_merge = (best_sim > self.merge_sim) & (sw.gather(-1, best_ix.unsqueeze(-1)).squeeze(-1) > 1e-6)
                    has_empty = sw <= 1e-6
                    first_empty = has_empty.float().argmax(dim=-1)
                    victim = sw.argmin(dim=-1)
                    target = torch.where(do_merge, best_ix, torch.where(has_empty.any(dim=-1), first_empty, victim))
                    t_one = F.one_hot(target, S).float().unsqueeze(-1) * write_ok[:, :, None, None]  # [B,H,S,1]
                    mrg = do_merge[:, :, None, None].float()
                    new_k = torch.where(mrg.bool(), F.normalize(0.7 * sk + 0.3 * k_star.unsqueeze(2), dim=-1), k_star.unsqueeze(2).expand(-1, -1, S, -1))
                    new_v = torch.where(mrg.bool(), 0.7 * sv + 0.3 * v_star.unsqueeze(2), v_star.unsqueeze(2).expand(-1, -1, S, -1))
                    sk = sk * (1 - t_one) + new_k * t_one
                    sv = sv * (1 - t_one) + new_v * t_one
                    new_w = torch.where(do_merge.unsqueeze(-1), torch.maximum(sw, b_star.unsqueeze(-1)), b_star.unsqueeze(-1).expand(-1, -1, S))
                    sw = sw * (1 - t_one.squeeze(-1)) + new_w * t_one.squeeze(-1)
            read = torch.sigmoid(self.read_logit).view(1, H, 1, 1) * torch.cat(reads, dim=2)
            state = {"M": mem, "K": sk, "V": sv, "w": sw}
        read_out = self.mem_proj(read.to(x.dtype).transpose(1, 2).reshape(B, T, H * Dm))
        return y + read_out, state


class HippocampusBlock(SynapseBlock):
    def __init__(self, cfg, mlp_hidden=768):
        super().__init__(cfg, mlp_hidden)
        self.attn = HippocampusAttention(cfg)


class HippocampusTransformer(SynapseTransformer):
    """Gen-2 H1+: Synapse delta memory in all blocks + ONE hippocampal slot bank in the
    final block (biologically: a single hippocampus serving many cortical areas — and
    slots cost ~20 extra kernels per chunk, so 1x not 8x). Same param class (17.3M)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.blocks = nn.ModuleList(
            [SynapseBlock(cfg, mlp_hidden=768) for _ in range(cfg.n_layer - 1)]
            + [HippocampusBlock(cfg, mlp_hidden=768)]
        )
        self._init_weights(cfg.n_layer)


class CortexTransformer(SynapseTransformer):
    """H3 Architecture: Dual Working/Episodic Surprise-Delta Memory + Input-Anchored Weight-Tied Iterative Workspace."""

    ponder_steps = 3
    max_ponder = 8

    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.max_context = None
        n_mem = max(2, cfg.n_layer - 3)
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(SynapseBlock(cfg, mlp_hidden=1344) for _ in range(n_mem))
        self.anchor_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.step_emb = nn.Embedding(self.max_ponder, cfg.d_model)
        self.core = Block(cfg)
        self.ponder_gate = nn.Parameter(torch.full((self.max_ponder,), 0.5))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight
        self.register_buffer("rope", rope_table(cfg.block_size, cfg.d_model // cfg.n_head), persistent=False)
        self.register_buffer("local_mask", local_causal_mask(cfg.block_size, self.local_window), persistent=False)
        self._init_weights(n_mem + self.ponder_steps)
        nn.init.normal_(self.step_emb.weight, std=0.01)

    def forward_stream(self, idx, targets=None, mems=None):
        eot = idx == 0
        x = self.emb(idx)
        new = []
        for i, block in enumerate(self.blocks):
            x, m = block(x, self.rope, self.local_mask, eot, None if mems is None else mems[i])
            new.append(m)
        anchor = self.anchor_proj(x)
        z = x
        for k in range(self.ponder_steps):
            sk = self.step_emb.weight[min(k, self.max_ponder - 1)].view(1, 1, -1)
            scale = torch.sigmoid(self.ponder_gate[min(k, self.max_ponder - 1)])
            z = z + scale * (self.core(z + anchor + sk, self.rope) - (z + anchor + sk))
        logits = self.head(self.norm(z))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        return logits, loss, new


# every architecture takes a Config and returns (logits, loss) from forward(idx, targets).
# optional: max_context (None = unbounded), segments (consecutive windows per training sequence),
# forward_stream(idx, targets, mems) -> (logits, loss, mems) for evaluation with memory carried across windows
class HippocampusW3Attention(HippocampusAttention):
    """Write-selectivity lever: write the top-3 surprising tokens per micro-chunk
    instead of top-1, so multi-fact chunks stop losing facts (gap<=103 residual)."""
    n_write_per_chunk = 3


class HippocampusW3Block(SynapseBlock):
    def __init__(self, cfg, mlp_hidden=768):
        super().__init__(cfg, mlp_hidden)
        self.attn = HippocampusW3Attention(cfg)


class HippocampusW3Transformer(SynapseTransformer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.blocks = nn.ModuleList(
            [SynapseBlock(cfg, mlp_hidden=768) for _ in range(cfg.n_layer - 1)]
            + [HippocampusW3Block(cfg, mlp_hidden=768)]
        )
        self._init_weights(cfg.n_layer)


MODELS = {
    "transformer": Transformer,
    "memory": MemoryTransformer,
    "synapse": SynapseTransformer,
    "hippocampus": HippocampusTransformer,
    "hippocampus_w3": HippocampusW3Transformer,
    "cortex": CortexTransformer,
}
