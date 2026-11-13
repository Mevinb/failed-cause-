N = 32
ATTN = [i for i in range(N) if (i + 1) % 4 == 0]
REC = [i for i in range(N) if i not in ATTN]


def lmap(layers):
    return ",".join(map(str, layers))


def cfg(name, note="", **env):
    return {"name": name, "note": note, "env": {f"EXP_{k.upper()}": v for k, v in env.items()}}


def baseline():
    return [cfg("baseline", "stock behaviour")]


def rewire():
    out = baseline()
    for i in range(N):
        out.append(cfg(f"drop_{i}", f"drop {'attn' if i in ATTN else 'rec'} layer {i}", layer_map=lmap([j for j in range(N) if j != i])))
    for a, b in [(1, 2), (9, 10), (17, 18), (25, 26), (3, 4), (15, 16), (27, 28)]:
        order = list(range(N))
        order[a], order[b] = order[b], order[a]
        out.append(cfg(f"swap_{a}_{b}", "swap adjacent layers", layer_map=lmap(order)))
    for lo, hi in [(8, 23), (12, 19), (4, 27)]:
        out.append(cfg(f"reverse_{lo}_{hi}", "reverse a middle block", layer_map=lmap(list(range(lo)) + list(range(hi, lo - 1, -1)) + list(range(hi + 1, N)))))
    out.append(cfg("no_attn", "remove all 8 full-attention layers", layer_map=lmap(REC)))
    out.append(cfg("attn_only", "keep only the 8 full-attention layers", layer_map=lmap(ATTN)))
    out.append(cfg("no_attn_mid", "remove attention layers 11,15,19", layer_map=lmap([i for i in range(N) if i not in (11, 15, 19)])))
    order = list(range(N))
    for a in ATTN:
        order[a - 1], order[a] = order[a], order[a - 1]
    out.append(cfg("attn_phase_shift", "every attention layer moved one slot earlier", layer_map=lmap(order)))
    out.append(cfg("interleave_halves", "interleave first and second half 0,16,1,17,...", layer_map=lmap([x for p in zip(range(16), range(16, 32)) for x in p])))
    return out


def loops():
    out = baseline()
    for s in [4, 8, 12, 16, 20, 24]:
        blk = list(range(s, s + 4))
        for n in [2, 3]:
            m = list(range(s + 4)) + blk * (n - 1) + list(range(s + 4, N))
            out.append(cfg(f"loop{n}_{s}_{s+3}", f"block {s}-{s+3} run {n}x", layer_map=lmap(m)))
            extra = range(s + 4, s + 4 + 4 * (n - 1))
            out.append(cfg(f"loop{n}_{s}_{s+3}_damp", f"block {s}-{s+3} run {n}x, extra passes at 0.5", layer_map=lmap(m), layer_scale=",".join(f"{v}:0.5" for v in extra)))
    for layer, n in [(14, 2), (14, 4), (15, 2), (15, 4), (29, 3)]:
        m = list(range(layer + 1)) + [layer] * (n - 1) + list(range(layer + 1, N))
        out.append(cfg(f"rep{n}_L{layer}", f"single {'attn' if layer in ATTN else 'rec'} layer {layer} x{n}", layer_map=lmap(m)))
    out.append(cfg("loop2_8_23", "whole middle 8-23 run twice", layer_map=lmap(list(range(24)) + list(range(8, 24)) + list(range(24, N)))))
    return out


def feedback():
    # alpha is norm-relative: added vector = alpha * rms(dst input) * h_src(t-1)/rms(h_src(t-1))
    out = baseline()
    for src, dst in [(31, 0), (31, 8), (23, 4), (15, 0)]:
        for a in [0.02, 0.1, 0.3, -0.1]:
            out.append(cfg(f"fb_{src}_{dst}_{a}", f"loop: h{src}(t-1) -> layer {dst}, rel {a}", feedback=f"{src}:{dst}:{a}"))
    for src, dst in [(8, 24), (4, 27)]:
        for a in [0.1, 0.3, 1.0]:
            out.append(cfg(f"fb_{src}_{dst}_{a}", f"skip-forward: h{src}(t-1) -> layer {dst}, rel {a}", feedback=f"{src}:{dst}:{a}"))
    return out


def decay():
    out = baseline()
    for a in [0.25, 0.5, 0.75, 0.9, 1.1, 1.5, 2.0, 4.0]:
        out.append(cfg(f"decay_{a}", "scale DeltaNet log-decay in all recurrent layers", ssm_decay=str(a)))
    early, late = [i for i in REC if i < 11], [i for i in REC if i > 20]
    for a in [0.25, 2.0]:
        out.append(cfg(f"decay_early_{a}", "recurrent layers 0-10 only", ssm_decay=",".join(f"{i}:{a}" for i in early)))
        out.append(cfg(f"decay_late_{a}", "recurrent layers 21-31 only", ssm_decay=",".join(f"{i}:{a}" for i in late)))
    return out


def scale():
    out = baseline()
    for a in [0.8, 0.9, 1.1, 1.2]:
        out.append(cfg(f"scale_all_{a}", "all residual updates scaled", layer_scale=",".join(f"{i}:{a}" for i in range(N))))
    for a in [0.0, 0.5, 1.5, 2.0]:
        out.append(cfg(f"scale_attn_{a}", "full-attention layers scaled", layer_scale=",".join(f"{i}:{a}" for i in ATTN)))
    for a in [0.5, 1.5]:
        out.append(cfg(f"scale_rec_{a}", "recurrent layers scaled", layer_scale=",".join(f"{i}:{a}" for i in REC)))
    return out


SWEEPS = {"baseline": baseline, "rewire": rewire, "loops": loops, "feedback": feedback, "decay": decay, "scale": scale}
