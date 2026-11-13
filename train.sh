#!/usr/bin/env bash
# Train the baseline transformer on TinyStories.
#
#   bash train.sh                 start, or continue after an interruption
#   bash train.sh NAME [options]  use another run name / pass extra options to lab/train.py
#
# Ctrl+C stops it. A checkpoint is saved every 2000 steps, so running it again continues from there.
# Startup takes up to a minute while the model is compiled.
set -euo pipefail
cd "$(dirname "$0")"

name=${1:-baseline}
shift || true

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
if [ "$free" -lt 2600 ]; then
    echo "Only $free MiB of GPU memory is free; training needs about 2200 MiB." >&2
    echo "Close Forge / ComfyUI / llama-server and run this again." >&2
    exit 1
fi

# --compile builds a small C file against Python's headers; treating them as system headers
# hides a harmless gcc warning, and TORCH_LOGS hides a harmless note from the model compiler
python_include=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
export C_INCLUDE_PATH="$python_include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export TORCH_LOGS="${TORCH_LOGS:--inductor}"

# keep the laptop from going to sleep while training runs; the lock is dropped when training exits
# (setsid keeps Ctrl+C away from the lock holder, which watches this process instead)
if command -v systemd-inhibit >/dev/null; then
    setsid systemd-inhibit --what=sleep:idle --who=agi-exi --why="model training" tail --pid=$$ -f /dev/null &
fi

exec .venv/bin/python lab/train.py "$name" --compile --resume "$@"
