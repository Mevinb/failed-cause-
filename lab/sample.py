"""Generate text from a trained run.

usage: python lab/sample.py RUN [--prompt "Once upon a time"] [-n 200] [--temp 0.8] [--seed 0] [--cpu]
"""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import MODELS, Config

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/tinystories"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    meta = json.loads((DATA / "meta.json").read_text())
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    ck = torch.load(ROOT / "runs" / a.run / "last.pt", map_location=dev)
    model = MODELS[ck["arch"]](Config(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["model"])
    ids = torch.tensor([[meta["eot"]] + tok.encode(a.prompt).ids], device=dev)
    ids = model.generate(ids, a.n, temperature=a.temp, stop=meta["eot"], generator=torch.Generator(device=dev).manual_seed(a.seed))
    print(tok.decode(ids[0, 1:].tolist()))


if __name__ == "__main__":
    main()
