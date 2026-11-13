"""Summarize results/*.jsonl into results/SUMMARY.md."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
CATS = ["fact", "arith", "seq", "lang", "passkey"]


def load(name):
    p = R / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def fmt_ppl(x):
    return "ERR" if x is None else (f"{x:.3f}" if x < 1000 else f"{x:.3g}")


def sweep_table(rows, base_ppl):
    out = ["| config | note | ppl | Δppl% | acc | " + " | ".join(CATS) + " | freeform[0] |", "|" + "---|" * (6 + len(CATS))]
    for r in rows:
        ppl = r.get("ppl")
        d = "" if ppl is None or base_ppl is None else f"{100 * (ppl / base_ppl - 1):+.1f}"
        cat = r.get("acc_cat", {})
        ff = (r.get("freeform") or [r.get("probe_error", r.get("ppl_error", ""))])[0]
        ff = ff.replace("\n", "⏎").replace("|", "/")[:70]
        out.append(f"| {r['name']} | {r.get('note','')} | {fmt_ppl(ppl)} | {d} | {r.get('acc','')} | "
                   + " | ".join(str(cat.get(c, "")) for c in CATS) + f" | {ff} |")
    return "\n".join(out)


def state_table(rows):
    out = ["| name | mode | alpha | layers | hit | output |", "|---|---|---|---|---|---|"]
    for r in rows:
        lay = r.get("layers")
        lay = "" if lay is None else (f"{len(lay)} layers" if len(lay) > 3 else ",".join(map(str, lay)))
        o = (r.get("output") or r.get("error", "")).replace("\n", "⏎").replace("|", "/")[:90]
        out.append(f"| {r['name']} | {r.get('mode','')} | {r.get('alpha','')} | {lay} | {r.get('hit','')} | {o} |")
    return "\n".join(out)


def main():
    base = load("baseline")
    base_ppl = base[0]["ppl"] if base else None
    parts = [f"# Results summary\n\nbaseline ppl={base_ppl} (wikitext-2 test, 20x512 chunks; feedback sweep uses 4 chunks)\n"]
    for s in ["rewire", "loops", "decay", "scale", "feedback"]:
        rows = load(s)
        if rows:
            bp = next((r["ppl"] for r in rows if r["name"] == "baseline"), base_ppl)
            parts.append(f"## {s}\n\n" + sweep_table(rows, bp) + "\n")
    st = load("state")
    if st:
        parts.append("## state surgery\n\n" + state_table(st) + "\n")
    (R / "SUMMARY.md").write_text("\n".join(parts))
    print((R / "SUMMARY.md").read_text())


if __name__ == "__main__":
    main()
