#!/usr/bin/env bash
# Runs every sweep in sequence; resumable (finished configs are skipped).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for s in baseline rewire loops decay scale; do
  $PY harness/run.py "$s"
done
$PY harness/state_exps.py
$PY harness/run.py feedback
echo ALL_DONE
