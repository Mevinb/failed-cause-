"""Run one or more experiment configs: perplexity + probe suite + freeform samples.

usage: python harness/run.py SWEEP [--only NAME ...] [--chunks 20] [--no-ppl] [--no-probes]
Results append to results/<SWEEP>.jsonl (configs already present are skipped unless --force).
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import probes  # noqa: E402
import sweeps  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BIN = Path(os.environ.get("EXP_BIN_DIR", ROOT / "llama.cpp-exp/build/bin"))
MODEL = Path(os.environ.get("EXP_MODEL", ROOT / "models/Qwen3.5-9B-Q4_K_M.gguf"))
WIKI = Path(os.environ.get("EXP_WIKI", ROOT / "data/wikitext-2-raw/wiki.test.raw"))
NGL = os.environ.get("NGL", "30")
LOGS = ROOT / "results/logs"


def exp_env(cfg):
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXP_") or k in ("EXP_BIN_DIR", "EXP_MODEL", "EXP_WIKI")}
    env.update(cfg["env"])
    return env


def check_exp_lines(log_text, cfg):
    seen = set(re.findall(r"\[exp\] (EXP_[A-Z_]+)=", log_text))
    missing = set(cfg["env"]) - seen
    if missing:
        raise RuntimeError(f"{cfg['name']}: hooks not confirmed in log: {sorted(missing)}")


def run_ppl(cfg, chunks):
    log = LOGS / f"{cfg['name']}.ppl.log"
    t = time.time()
    p = subprocess.run(
        [str(BIN / "llama-perplexity"), "-m", str(MODEL), "-f", str(WIKI), "-c", "512", "-b", "512", "--chunks", str(chunks), "-ngl", NGL],
        env=exp_env(cfg), capture_output=True, text=True,
    )
    out = p.stdout + p.stderr
    log.write_text(out)
    check_exp_lines(out, cfg)
    m = re.search(r"Final estimate: PPL = ([0-9.naNinf+-]+) \+/- ([0-9.naNinf]+)", out)
    if p.returncode != 0 or not m:
        return {"ppl": None, "ppl_err": None, "ppl_error": f"rc={p.returncode} see {log.name}", "ppl_s": time.time() - t}
    return {"ppl": float(m.group(1)), "ppl_err": float(m.group(2)), "ppl_s": round(time.time() - t, 1)}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["content"]


def run_probes(cfg):
    log = LOGS / f"{cfg['name']}.server.log"
    port = free_port()
    with open(log, "w") as lf:
        srv = subprocess.Popen(
            [str(BIN / "llama-server"), "-m", str(MODEL), "-ngl", NGL, "-c", "4096", "-np", "1", "--port", str(port)],
            env=exp_env(cfg), stdout=lf, stderr=subprocess.STDOUT,
        )
    t = time.time()
    try:
        while True:
            if srv.poll() is not None:
                return {"probe_error": f"server exited rc={srv.returncode} see {log.name}"}
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if time.time() - t > 300:
                return {"probe_error": "server start timeout"}
            time.sleep(1)
        check_exp_lines(log.read_text(), cfg)
        per_cat, rows = defaultdict(list), []
        for it in probes.build():
            out = post(port, {"prompt": it["prompt"], "n_predict": it["n"], "temperature": 0, "top_k": 1, "cache_prompt": False, "stop": ["\nQ:"]})
            ok = probes.score(out, it["answers"])
            per_cat[it["cat"]].append(ok)
            rows.append({"id": it["id"], "ok": ok, "out": out[:80]})
        free = [post(port, {"prompt": p, "n_predict": 80, "temperature": 0, "top_k": 1, "cache_prompt": False}) for p in probes.FREEFORM]
        acc = {c: round(sum(v) / len(v), 3) for c, v in per_cat.items()}
        total = sum(sum(v) for v in per_cat.values()) / sum(len(v) for v in per_cat.values())
        return {"acc": round(total, 3), "acc_cat": acc, "probes": rows, "freeform": free, "probe_s": round(time.time() - t, 1)}
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--chunks", type=int, default=None, help="default: 20, or 4 for the feedback sweep (1-token ubatches)")
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.chunks is None:
        a.chunks = 4 if a.sweep == "feedback" else 20
    LOGS.mkdir(parents=True, exist_ok=True)
    out = ROOT / f"results/{a.sweep}.jsonl"
    done = set()
    if out.exists() and not a.force:
        done = {json.loads(l)["name"] for l in out.read_text().splitlines() if l.strip()}
    for cfg in sweeps.SWEEPS[a.sweep]():
        if (a.only and cfg["name"] not in a.only) or cfg["name"] in done:
            continue
        row = {"name": cfg["name"], "env": cfg["env"], "note": cfg.get("note", ""), "chunks": a.chunks}
        if not a.no_ppl:
            row.update(run_ppl(cfg, a.chunks))
        if not a.no_probes:
            row.update(run_probes(cfg))
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{cfg['name']:<28} ppl={row.get('ppl')} acc={row.get('acc')} {row.get('acc_cat', row.get('probe_error', ''))}", flush=True)


if __name__ == "__main__":
    main()
