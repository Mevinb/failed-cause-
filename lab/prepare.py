"""Train a byte-level BPE tokenizer on TinyStories and encode train/valid into uint16 token streams.

usage: python lab/prepare.py [--vocab 8192]
Writes data/tinystories/{tokenizer.json,train.bin,val.bin,meta.json}.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA = Path(__file__).resolve().parent.parent / "data/tinystories"
EOT = "<|endoftext|>"


def stories(path, chunk_chars=1 << 24):
    buf = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while chunk := f.read(chunk_chars):
            *done, buf = (buf + chunk).split(EOT)
            yield from (s.strip() for s in done if s.strip())
    if buf.strip():
        yield buf.strip()


def train_tokenizer(vocab):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOT], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    # every 4th story is plenty to learn the merges
    tok.train_from_iterator(itertools.islice(stories(DATA / "TinyStories-train.txt"), 0, None, 4), trainer)
    tok.save(str(DATA / "tokenizer.json"))
    return tok


def encode(tok, src, dst):
    eot = tok.token_to_id(EOT)
    n_tok, n_story = 1, 0
    with open(dst, "wb") as out:
        np.array([eot], dtype=np.uint16).tofile(out)
        for group in itertools.batched(stories(src), 20000):
            arr = np.concatenate([np.array(e.ids + [eot], dtype=np.uint16) for e in tok.encode_batch(list(group))])
            arr.tofile(out)
            n_tok, n_story = n_tok + len(arr), n_story + len(group)
    return n_tok, n_story


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    a = ap.parse_args()
    assert a.vocab <= 65536, "token ids are stored as uint16"
    t = time.time()
    tok = train_tokenizer(a.vocab)
    print(f"tokenizer: vocab={tok.get_vocab_size()} ({time.time() - t:.0f}s)", flush=True)
    meta = {"vocab_size": tok.get_vocab_size(), "eot": tok.token_to_id(EOT)}
    for split, src in [("val", "TinyStories-valid.txt"), ("train", "TinyStories-train.txt")]:
        n_tok, n_story = encode(tok, DATA / src, DATA / f"{split}.bin")
        meta[split] = {"tokens": n_tok, "stories": n_story}
        print(f"{split}: {n_story} stories, {n_tok} tokens ({time.time() - t:.0f}s)", flush=True)
    (DATA / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
