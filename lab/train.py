"""Train a small language model on TinyStories.

usage: python lab/train.py NAME [--steps N] [--arch transformer] [--compile] [--resume]
Writes runs/NAME/{log.jsonl,last.pt}. Every architecture is trained and scored the same way,
so runs with the same --steps/--batch/--block-size are directly comparable.
"""
import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model import MODELS, Config

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/tinystories"


def mmss(seconds):
    return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"


def save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--arch", default="transformer", choices=sorted(MODELS))
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-batches", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    out = ROOT / "runs" / a.name
    if (out / "last.pt").exists() and not a.resume:
        sys.exit(f"{out} already has a checkpoint; pass --resume or pick a new name")
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((DATA / "meta.json").read_text())
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    train = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    T, B, dev = a.block_size, a.batch, "cuda"

    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = Config(vocab_size=meta["vocab_size"], block_size=T, n_layer=a.n_layer, n_head=a.n_head, d_model=a.d_model)
    raw = MODELS[a.arch](cfg).to(dev)
    groups = [{"params": [p for p in raw.parameters() if p.dim() >= 2], "weight_decay": 0.1},
              {"params": [p for p in raw.parameters() if p.dim() < 2], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), fused=True)
    start = 0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
    if start >= a.steps:
        print(f"{a.name} has already finished its {a.steps} steps")
        return
    model = torch.compile(raw) if a.compile else raw
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    # models with memory train on `segments` consecutive windows per sequence; tokens per step stay B*T
    segs = getattr(raw, "segments", 1)
    if B % segs:
        sys.exit(f"--batch {B} must be divisible by the model's {segs} segments")
    streaming = hasattr(raw, "forward_stream")

    def batch(data, offsets, length=T):
        x = torch.from_numpy(np.stack([data[o:o + length + 1] for o in offsets]).astype(np.int64)).to(dev)
        return x[:, :-1], x[:, 1:]

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        p = (step - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

    @torch.no_grad()
    def evaluate(n_batches, stream):
        # the same fixed, non-overlapping windows from the start of the validation stream for every model.
        # stream=True: B lanes of n consecutive windows each, memory carried from window to window inside a lane.
        model.eval()
        n = min(n_batches, (len(val) - 1) // (T * B))
        tot, mems = 0.0, None
        for i in range(n):
            with autocast:
                if stream:
                    x, y = batch(val, [(j * n + i) * T for j in range(B)])
                    _, loss, mems = raw.forward_stream(x, y, mems)
                else:
                    x, y = batch(val, [(i * B + j) * T for j in range(B)])
                    loss = model(x, y)[1]
            tot += loss.item()
        model.train()
        return tot / n, n * B * T

    def sample():
        ids = torch.tensor([[meta["eot"]] + tok.encode("Once upon a time").ids], device=dev)
        with autocast:
            ids = raw.generate(ids, 100, stop=meta["eot"], generator=torch.Generator(device=dev).manual_seed(0))
        return tok.decode(ids[0, 1:].tolist())

    log = open(out / "log.jsonl", "a")

    def emit(row, text):
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(text, flush=True)

    n_params = sum(p.numel() for p in raw.parameters())
    emit({"event": "resume" if start else "start", "arch": a.arch, "params": n_params, "cfg": asdict(cfg), "args": vars(a),
          "tokens_per_step": B * T, "train_tokens": len(train)},
         f"{a.name}: {a.arch}, {n_params / 1e6:.1f}M parameters, steps {start}-{a.steps} x {B * T} tokens "
         f"({a.steps * B * T / 1e6:.0f}M tokens in total, {a.steps * B * T / len(train):.2f} passes over the data)")

    t0 = t_log = time.time()
    acc, n_acc = torch.zeros((), device=dev), 0
    for step in range(start, a.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch(train, np.random.default_rng([a.seed, step]).integers(0, len(train) - segs * T - 1, B // segs), segs * T)
        with autocast:
            loss = model(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        opt.step()
        acc += loss.detach()
        n_acc += 1
        done = step + 1

        if done % a.log_every == 0:
            train_loss = acc.item() / n_acc
            if not math.isfinite(train_loss):
                sys.exit(f"step {done}: training loss is not finite")
            now = time.time()
            tok_s = n_acc * B * T / (now - t_log)
            left = (a.steps - done) * (now - t0) / (done - start)
            emit({"step": done, "train_loss": round(train_loss, 4), "lr": round(lr, 6), "tok_s": round(tok_s),
                  "elapsed_s": round(now - t0), "vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)},
                 f"step {done:>{len(str(a.steps))}}/{a.steps}  loss {train_loss:.3f}  lr {lr:.1e}  {tok_s / 1000:.0f}k tok/s  "
                 f"elapsed {mmss(now - t0)}  left ~{mmss(left)}")
            acc.zero_()
            n_acc, t_log = 0, now

        if done % a.eval_every == 0 or done == a.steps:
            final = done == a.steps
            n_eval = 10**9 if final else a.eval_batches
            val_loss, val_tokens = evaluate(n_eval, streaming)
            row = {"step": done, "val_loss": round(val_loss, 4), "val_ppl": round(math.exp(val_loss), 3),
                   "val_tokens": val_tokens, "final": final}
            extra = ""
            if streaming and final:
                # same windows with the memory wiped before each one: how much of the score comes from the memory
                nomem = evaluate(n_eval, False)[0]
                row["val_loss_nomem"] = round(nomem, 4)
                extra = f"; with the memory wiped before every window: {nomem:.4f}"
            save({"model": raw.state_dict(), "opt": opt.state_dict(), "step": done, "arch": a.arch, "cfg": asdict(cfg)}, out / "last.pt")
            story = sample()
            emit({**row, "elapsed_s": round(time.time() - t0), "sample": story},
                 f"\n--- step {done}: {'FINAL ' if final else ''}validation loss {val_loss:.4f} (perplexity {math.exp(val_loss):.2f}){extra}, "
                 f"checkpoint saved. Sample:\n{story}\n---\n")
            t_log = time.time()
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped. Run the same command again to continue from the last checkpoint.")
