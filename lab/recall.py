"""Long-range recall test: a fact early in a story, filler text of a set length, then a cue whose next word is the fact.

  python lab/recall.py build          write data/recall_test.jsonl (fixed seed; every model gets the same items)
  python lab/recall.py eval RUN       score runs/RUN/last.pt, write runs/RUN/recall.json

Each item is forced choice between 4 single-token candidates (chance 25%). Every base item appears 4 times, once with
each candidate as the right answer, so a model's preference for a particular word cannot raise its score.
Models that can only see the last N tokens (attribute `max_context`, default cfg.block_size) get the input cropped
from the left, so facts further back than their window are simply gone.
"""
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import MODELS, Config

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/tinystories"
TEST = ROOT / "data/recall_test.jsonl"

DISTANCES = [0, 32, 96, 192, 400, 800]
BASE_PER_CELL = 60
PEOPLE = ["Lily", "Tom", "Ben", "Mia", "Sam", "Sue", "Anna", "Jack"]
PETS = ["Max", "Buddy", "Coco", "Rex", "Daisy", "Lucky", "Rosie", "Pip", "Teddy", "Sparky", "Milo", "Ruby"]
COLORS = ["red", "blue", "green", "yellow", "pink", "purple", "orange", "brown"]
OBJECTS = ["ball", "kite", "hat", "box", "car", "cup"]
TASKS = {
    "pet": ("{p} had a little dog named {x}.", '{p} looked around and called, "Come here,', PETS),
    "color": ("{p} had a {x} {o}.", "{p} picked up the", COLORS),
}
BANNED = re.compile(r"\b(" + "|".join(PEOPLE + PETS + COLORS + OBJECTS + ["dog", "puppy", "pet"]) + r")\b", re.I)


def filler_sentences():
    text = (DATA / "TinyStories-valid.txt").read_text(encoding="utf-8")
    stories = []
    for s in text.split("<|endoftext|>"):
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", s.replace("\n", " ")) if x.strip()]
        sents = [x for x in sents if not BANNED.search(x) and '"' not in x]
        if sents:
            stories.append(sents)
    return stories


def make_filler(stories, tok, target, rng):
    if target == 0:
        return "", 0
    out, n = [], 0
    i = rng.randrange(len(stories))
    while n < target:
        for s in stories[i]:
            out.append(s)
            n = len(tok.encode(" " + " ".join(out)).ids)
            if n >= target:
                break
        i = (i + 1) % len(stories)
    return " " + " ".join(out), n


def build():
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    for w in PEOPLE + PETS + COLORS:
        assert len(tok.encode(" " + w).ids) == 1, f"{w} is not a single token"
    rng = random.Random(1234)
    stories = filler_sentences()
    items = []
    for task, (fact, cue, pool) in TASKS.items():
        for d in DISTANCES:
            for b in range(BASE_PER_CELL):
                p, o = rng.choice(PEOPLE), rng.choice(OBJECTS)
                cands = rng.sample(pool, 4)
                filler, n_fill = make_filler(stories, tok, d, rng)
                for k, x in enumerate(cands):
                    items.append({"task": task, "distance": d, "filler_tokens": n_fill, "base": b, "answer": k, "candidates": cands,
                                  "text": fact.format(p=p, x=x, o=o) + filler + " " + cue.format(p=p)})
    with open(TEST, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} items to {TEST}")


@torch.no_grad()
def evaluate(run):
    meta = json.loads((DATA / "meta.json").read_text())
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    ck = torch.load(ROOT / "runs" / run / "last.pt", map_location="cuda")
    model = MODELS[ck["arch"]](Config(**ck["cfg"])).cuda().eval()
    model.load_state_dict(ck["model"])
    ctx = getattr(model, "max_context", model.cfg.block_size)

    items = [json.loads(l) for l in TEST.read_text().splitlines()]
    seqs, cropped = [], []
    for it in items:
        ids = [meta["eot"]] + tok.encode(it["text"]).ids
        cropped.append(ctx is not None and len(ids) > ctx)
        seqs.append(ids[-ctx:] if ctx is not None else ids)
    cand_ids = [[tok.encode(" " + c).ids[0] for c in it["candidates"]] for it in items]

    correct = [False] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(seqs[i]))
    s = 0
    while s < len(order):
        # keep the full logits tensor (batch x length x vocab) around 16k positions
        bs = min(64, max(4, 16384 // len(seqs[order[min(s + 63, len(order) - 1)]])))
        idx = order[s:s + bs]
        s += bs
        L = max(len(seqs[i]) for i in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :len(seqs[i])] = torch.tensor(seqs[i])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.cuda())[0].float()
        for r, i in enumerate(idx):
            lp = F.log_softmax(logits[r, len(seqs[i]) - 1], -1)[cand_ids[i]]
            correct[i] = int(lp.argmax()) == items[i]["answer"]

    cells = defaultdict(list)
    for it, ok, cr in zip(items, correct, cropped):
        cells[(it["task"], it["distance"])].append((ok, cr, it["filler_tokens"]))
        cells[("all", it["distance"])].append((ok, cr, it["filler_tokens"]))
    res = {"run": run, "arch": ck["arch"], "step": ck["step"], "max_context": ctx, "cells": []}
    for (task, d), v in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        n, acc = len(v), sum(o for o, _, _ in v) / len(v)
        res["cells"].append({"task": task, "distance": d, "n": n, "acc": round(acc, 4),
                             "ci95": round(1.96 * math.sqrt(acc * (1 - acc) / n), 4),
                             "cropped": round(sum(c for _, c, _ in v) / n, 3),
                             "mean_filler_tokens": round(sum(f for _, _, f in v) / n, 1)})
    (ROOT / "runs" / run / "recall.json").write_text(json.dumps(res, indent=1))

    window = "no context limit" if ctx is None else f"sees last {ctx} tokens"
    print(f"{run} ({ck['arch']}, step {ck['step']}, {window}). Chance = 25%.")
    print(f"{'gap (tokens)':>12} | {'pet name':>14} | {'color':>14} | {'both':>14} | fact cut off")
    by = {(c["task"], c["distance"]): c for c in res["cells"]}
    for d in DISTANCES:
        row = [f"{100 * by[(t, d)]['acc']:5.1f}% ±{100 * by[(t, d)]['ci95']:4.1f}" for t in ("pet", "color", "all")]
        print(f"{by[('all', d)]['mean_filler_tokens']:>12.0f} | {row[0]:>14} | {row[1]:>14} | {row[2]:>14} | {100 * by[('all', d)]['cropped']:.0f}%")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["eval"] and len(sys.argv) == 3:
        evaluate(sys.argv[2])
    else:
        sys.exit(__doc__)
