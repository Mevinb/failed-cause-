"""Recurrent-state surgery experiments via llama-exp-state.

usage: python harness/state_exps.py [--only GROUP ...]
Results append to results/state.jsonl.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import BIN, MODEL, NGL, ROOT  # noqa: E402
from sweeps import ATTN, REC  # noqa: E402

FACT = "Note to self: my name is Zorbak, my favourite colour is teal, and the secret code is 7391. "
NEUTRAL = "Today the weather is mild and the streets are quiet. People walk their dogs in the park. "
Q_CODE = "\nQ: What is the secret code?\nA:"
Q_NAME = "\nQ: What is my name?\nA:"
FRENCH = "Bonjour, je m'appelle Marie. J'habite à Paris et j'aime beaucoup le fromage et le vin rouge. "
ANGRY = "I am furious. Everything is terrible, I hate this stupid broken day and everyone in it. "
JOYFUL = "I am so happy! Everything is wonderful, I love this beautiful bright day and everyone in it. "
Q_WRITE = "\nWrite one sentence about the weather:"

EARLY = [i for i in REC if i < 11]
MID = [i for i in REC if 11 <= i < 22]
LATE = [i for i in REC if i >= 22]


def group_recall():
    out = []
    for q, key, qn in [(Q_CODE, "7391", "code"), (Q_NAME, "zorbak", "name")]:
        out.append(dict(name=f"recall_{qn}_control_has_fact", donor=NEUTRAL, recipient=FACT + NEUTRAL, query=q, mode="blend", alpha=0.0, key=key))
        out.append(dict(name=f"recall_{qn}_control_no_fact", donor=NEUTRAL, recipient=NEUTRAL, query=q, mode="blend", alpha=0.0, key=key))
        for lname, layers in [("all", REC), ("early", EARLY), ("mid", MID), ("late", LATE)]:
            out.append(dict(name=f"recall_{qn}_transplant_{lname}", donor=FACT + NEUTRAL, recipient=NEUTRAL, query=q, mode="transplant", layers=layers, key=key))
        for a in [0.5, 1.5, 3.0, -1.0]:
            out.append(dict(name=f"recall_{qn}_blend_{a}", donor=FACT + NEUTRAL, recipient=NEUTRAL, query=q, mode="blend", alpha=a, key=key))
        for i in REC:
            out.append(dict(name=f"recall_{qn}_transplant_L{i}", donor=FACT + NEUTRAL, recipient=NEUTRAL, query=q, mode="transplant", layers=[i], key=key))
    return out


def group_dissociate():
    out = []
    for q, key, qn in [(Q_CODE, "7391", "code"), (Q_NAME, "zorbak", "name")]:
        for lname, layers in [("all", REC), ("early", EARLY), ("mid", MID), ("late", LATE)]:
            out.append(dict(name=f"dissoc_{qn}_zero_{lname}", donor=NEUTRAL, recipient=FACT + NEUTRAL, query=q, mode="zero", layers=layers, key=key))
            out.append(dict(name=f"dissoc_{qn}_freeze_{lname}", donor=NEUTRAL, recipient=FACT + NEUTRAL, query=q, mode="freeze", layers=layers, key=key))
        out.append(dict(name=f"dissoc_{qn}_overwrite_neutral", donor=NEUTRAL, recipient=FACT + NEUTRAL, query=q, mode="transplant", layers=REC, key=key))
    return out


def group_style():
    out = []
    for dname, donor in [("french", FRENCH), ("angry", ANGRY), ("joyful", JOYFUL)]:
        for a in [0.0, 0.5, 1.0, 2.0, 4.0]:
            out.append(dict(name=f"style_{dname}_blend_{a}", donor=donor, recipient=NEUTRAL, query=Q_WRITE, mode="blend", alpha=a, n=48))
        out.append(dict(name=f"style_{dname}_diff_2.0", donor=donor, recipient=NEUTRAL, query=Q_WRITE, mode="blend", alpha=-2.0, n=48, note="negative: push away from donor"))
    return out


GROUPS = {"recall": group_recall, "dissociate": group_dissociate, "style": group_style}


def run_one(e, tmp):
    files = {}
    for k in ("donor", "recipient", "query"):
        p = Path(tmp) / f"{k}.txt"
        p.write_text(e[k])
        files[k] = str(p)
    cmd = [str(BIN / "llama-exp-state"), "-m", str(MODEL), "-ngl", NGL, "-c", "2048", "-ub", "128", "--donor", files["donor"], "--recipient", files["recipient"],
           "--query", files["query"], "--mode", e["mode"], "-n", str(e.get("n", 12))]
    if "alpha" in e:
        cmd += ["--alpha", str(e["alpha"])]
    if "layers" in e:
        cmd += ["--layers", ",".join(map(str, e["layers"]))]
    p = subprocess.run(cmd, capture_output=True, text=True, env={k: v for k, v in os.environ.items() if not k.startswith("EXP_") or k in ("EXP_BIN_DIR", "EXP_MODEL")})
    js = [l for l in p.stdout.splitlines() if l.startswith("{") and '"output"' in l]
    if p.returncode != 0 or not js:
        return {"error": f"rc={p.returncode}", "stderr_tail": p.stderr[-800:]}
    res = json.loads(js[-1])
    if "key" in e:
        res["hit"] = e["key"] in res["output"].lower()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = ROOT / "results/state.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["name"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    with tempfile.TemporaryDirectory() as tmp:
        for g, fn in GROUPS.items():
            if a.only and g not in a.only:
                continue
            for e in fn():
                if e["name"] in done:
                    continue
                r = {"name": e["name"], "group": g, **{k: v for k, v in e.items() if k in ("mode", "alpha", "layers", "note")}, **run_one(e, tmp)}
                with open(out, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"{e['name']:<36} hit={r.get('hit')} out={r.get('output', r.get('error'))!r:.90}", flush=True)


if __name__ == "__main__":
    main()
