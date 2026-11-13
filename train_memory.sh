#!/usr/bin/env bash
# Train the memory transformer (lab/model.py: MemoryTransformer) with exactly the baseline's budget,
# then score it on the recall test next to the baseline.
#
#   bash train_memory.sh      start, or continue after an interruption (Ctrl+C is safe)
set -euo pipefail
cd "$(dirname "$0")"

bash train.sh memory --arch memory

cd lab
../.venv/bin/python recall.py eval memory
[ -f ../runs/baseline/last.pt ] && ../.venv/bin/python recall.py eval baseline
