# agi-exi — AGI research laboratory (Qwen3.5-9B surgery + from-scratch architecture lab)

**START HERE: `DOCUMENTATION.md`** — complete plain-English + technical guide to everything
built, tested, discovered, and failed. `research/MASTER_EXPERIMENT_LOG.md` — chronological
lab notebook of every experiment. `research/` — literature DB, architecture spec, benchmark
reports, hypotheses registry, roadmap, progress reports.

This is a long-running autonomous AGI research program. Every experiment (including
negative results and bugs) is permanently logged in `research/MASTER_EXPERIMENT_LOG.md`
and summarized in `DOCUMENTATION.md`. Never delete or fabricate results.

Experimental fork of llama.cpp (base `bebc935`) in worktree
`/home/mevlec/Data/exp/agi-exi/llama.cpp-exp` (branch `agi-exp`).
Upstream `/home/mevlec/llama.cpp` and its `build/` are NOT touched. All changes uncommitted.

## Build

```sh
cd llama.cpp-exp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target llama-perplexity llama-completion llama-cli llama-server llama-exp-state
```

## Binaries

- `llama.cpp-exp/build/bin/llama-perplexity`
- `llama.cpp-exp/build/bin/llama-completion`, `llama-cli`, `llama-server`
- `llama.cpp-exp/build/bin/llama-exp-state`
- stock reference: `/home/mevlec/llama.cpp/build/bin/llama-perplexity`

## Model / data

- Model: `models/Qwen3.5-9B-Q4_K_M.gguf` (arch `qwen35`, 32 blocks; layers 3,7,...,31 full attention, rest Gated-DeltaNet)
- Data: `data/wikitext-2-raw/wiki.test.raw` (via llama.cpp `scripts/get-wikitext-2.sh`)
- Test texts for llama-exp-state: `data/{donor_same,donor_diff,recipient,query}.txt`

## GPU offload

RTX 4050 6 GB. Largest working `-ngl` at `-c 2048` is 31 (`-ngl 32` fails: output tensor
+ compute buffers no longer fit). `-ngl 31` is borderline and depends on other GPU
consumers; `-ngl 30` is the stable choice. PPL does not depend on `-ngl`.

Reference perplexity run (stock == patched == 8.3730):

```sh
llama-perplexity -m models/Qwen3.5-9B-Q4_K_M.gguf -c 512 --chunks 20 -ngl 30 \
    -f data/wikitext-2-raw/wiki.test.raw -s 1     # ~15 s
```

## Env-var hooks

Parsed once per process (static). Each prints `[exp] NAME=<value>` to stderr.
Malformed values abort. Empty/unset = stock behavior (verified bit-identical PPL).

| Var | Format | Semantics |
|---|---|---|
| `EXP_LAYER_MAP` | `0-31` / `0-31,28-31` / `0-4,6-31` | virtual layer i uses weights+type of source map[i]; shared tensors (no copy); each virtual layer gets its own recurrent/KV slot; `n_layer = len(map)` |
| `EXP_LAYER_SCALE` | `i:alpha,...` (virtual idx) | both residual updates of layer i scaled: `x += alpha*mixer`, `x += alpha*ffn` |
| `EXP_FEEDBACK` | `src:dst:alpha[:raw]` | adds `alpha * rms(dst input) * h_src(t-1)/rms(h_src(t-1))` to residual entering dst (norm-relative); with `:raw` adds `alpha*h_src(t-1)` unscaled; persistent carry across ubatches; resets on position discontinuity; single seq only |
| `EXP_SSM_DECAY` | `alpha` or `i:alpha,...` | scales the GDN log-decay gate (softplus(alpha+dt)*ssm_a) before exp; <1 slower forgetting, >1 faster |

Notes:
- `EXP_FEEDBACK` with `dst <= src` is a cross-token loop: `n_ubatch` is clamped to 1
  (warns). `dst > src` shifts within the ubatch (carry + concat). Requires a single
  sequence in the batch — llama-perplexity needs `-b <= -c` (e.g. `-b 512`) since it
  otherwise splits into `n_batch/n_ctx` parallel seqs.
- `EXP_LAYER_MAP` source indices are source (GGUF) layer ids; SCALE/FEEDBACK/DECAY
  indices are virtual. Dropping an attention layer leaves a model with fewer KV
  layers; dropping all attention or all recurrent layers is untested.

## llama-exp-state

```sh
llama-exp-state -m MODEL -ngl N -c 2048 -ub 128 \
    --donor FILE --recipient FILE --query FILE \
    [--layers LIST] [--mode transplant|blend|zero|freeze|none] [--alpha A] [-n 64] \
    [--chat] [--dump-state FILE]
```

Processes donor in context A, recipient in fresh context B, edits B's per-layer
recurrent state (conv `r_l` + ssm `s_l` rows, attention KV untouched), processes the
query, greedy-generates `-n` tokens, prints text + JSON line. `--dump-state` writes
per-layer L2 norms of conv/ssm state after the recipient pass. Modes:
`transplant` B:=donor, `blend` B:=(1-a)B+a*donor, `zero` B:=0, `freeze` restores the
post-recipient state before every subsequent decode step, `none` = baseline.

Two contexts share the model but each reserves compute buffers — use `-ub 128` (or
lower `-ngl`) on the 6 GB card.

## Logs

`data/logs/ppl_*.log` — `/usr/bin/time -v` wrapped perplexity runs.

## Sweeps (harness/)

- `bash harness/run_all.sh` runs every sweep (rewire, loops, decay, scale, state surgery, feedback). Resumable:
  configs already in `results/<sweep>.jsonl` are skipped. One config = perplexity + a 42-question probe suite (~1 min).
- `.venv/bin/python harness/analyze.py` writes `results/SUMMARY.md`.
- Never run it next to another GPU job: a second llama process makes model loading fail.

## From-scratch training lab (lab/, train.sh)

Separate from the Qwen work: small models trained from scratch on TinyStories, to compare architectures
against a baseline transformer under one fixed budget.

- Env: the same `.venv` (py3.12) with `torch==2.11.0` (cu130) and `tokenizers==0.23.3`:
  `VIRTUAL_ENV=.venv uv pip install --exclude-newer <date 7+ days ago> torch==2.11.0 tokenizers==0.23.3`.
- Data: `data/tinystories/TinyStories-{train,valid}.txt` (HF dataset `roneneldan/TinyStories`).
  `.venv/bin/python lab/prepare.py` (~3 min) trains a byte-level BPE (vocab 8192, `<|endoftext|>` = id 0) and writes
  `tokenizer.json`, `train.bin` (458.3M tokens, uint16), `val.bin` (4.61M tokens), `meta.json`.
- Train: `bash train.sh [NAME] [lab/train.py options]` (default run `baseline`). It resumes when `runs/NAME/last.pt`
  exists, refuses to start with under 2600 MiB of free VRAM, and holds a sleep inhibitor. Output: `runs/NAME/{log.jsonl,last.pt}`.
- Standard budget, keep it identical when comparing architectures: 24000 steps x 32 sequences x 256 tokens
  (197M tokens, 0.43 passes), lr 6e-4 cosine to 10%, warmup 500, AdamW, bf16 autocast. Score = final `val_loss` in
  `log.jsonl` (whole validation set).
- Baseline `transformer`: 8 layers, d=384, 6 heads, RoPE, RMSNorm, SwiGLU, tied embeddings, 17.3M parameters.
- Baseline result (`runs/baseline`, 2026-11-12, seed 0): final val_loss **1.5888** (ppl 4.898) on the whole validation set;
  1.6408 on the first 100 batches (the subset used for the step-2000..22000 checkpoints, so compare those against 1.6408).
  Took 33.5 min at ~101k tok/s, 1.66 GB peak VRAM. Loss was still falling at the end (not converged).
- Measured on the RTX 4050: 88k tok/s with `--compile` (65k without), 2.1 GB VRAM, 2.5 GB RAM, about 40 min per run.
  Batch 64x512 does not fit in 6 GB (64x256 or 32x512 need 4.2 GB uncompiled).
- New architecture: add a class to `MODELS` in `lab/model.py` (constructor takes `Config`; `forward(idx, targets)`
  returns `(logits, loss)`; `generate` as in `Transformer`), then `bash train.sh NAME --arch KEY`.
- Sample: `.venv/bin/python lab/sample.py NAME [--prompt TEXT] [-n 200] [--temp 0.8] [--cpu]`.
- Recall test (`lab/recall.py`, run from `lab/`): `../.venv/bin/python recall.py eval NAME` writes `runs/NAME/recall.json`.
  Items are fixed in `data/recall_test.jsonl` (`recall.py build`, seed 1234; don't rebuild between comparisons).
  Fact early, filler of 0/32/96/192/400/800 tokens, cue; 4-way forced choice on single tokens, chance 25%, every base item
  rotated through all 4 answers. 60 base items x 4 per task and gap. The printed ±CI treats the 4 rotations as
  independent, so it is too narrow; for significance count base items. Archs can set `max_context` (None = unbounded);
  default crops input to `cfg.block_size`.
- Baseline recall (both tasks): gap 0: 97.7%, 38: 84.2%, 103: 83.1%, 198: 56.0%, 406 and 807: 25.0% (fact outside the
  256-token window).
- `memory` arch (`MemoryTransformer`, start with `bash train_memory.sh`): the baseline plus, in every attention layer, a
  per-head 64x64 associative memory carried across 256-token windows. Read by every query (L2-normalised q, learned
  per-head read gate initialised at 0). Written once per window: per-token write strength sigmoid(Linear(x)) (bias -1),
  one delta-rule step on the window's normalised keys/values divided by (1 + sum of write strengths), learned per-head
  decay (init sigmoid(3) = 0.95). Memory math runs in fp32 under autocast. +18.6k parameters (+0.11%). Trains on 8
  sequences x 4 consecutive windows per step (same 8192 tokens/step), gradients through the memory. Closest prior work:
  Infini-attention (delta variant), Recurrent Memory Transformer, DeltaNet.
- Memory-model evaluation: validation is streamed in 32 lanes of consecutive windows with memory carried (same windows
  and targets as the baseline); the final eval also logs `val_loss_nomem` with memory wiped before every window.
  Checked: read gate 0 gives outputs identical to the baseline; whole-sequence forward == window-by-window streaming;
  memory norm stays bounded over 200 windows. Smoke run: same 87k tok/s, 2.0 GB VRAM, ~1 min compile at the start.
- Verify after code changes: `bash train.sh smoke --steps 300 --eval-every 150 --log-every 50 --eval-batches 20`
  (about 1 min; loss should fall below 4 and the sample should look like a story), then delete `runs/smoke`.
- Iso-parameter (`17.3M ± 0.28%`) memory/workspace architectures added to `lab/model.py`:
  - `synapse` (`SynapseTransformer`): local sliding-window working attention (`W=64`) + decoupled position-invariant Q/K/V, micro-chunk (`C=32`) surprise-gated Delta fast-weight memory with sphere-normalized values and exact `<|endoftext|>` (`id=0`) episode resets. At 4000 steps (`runs/synapse_v2_4k`), lowers whole-val loss from `1.9029` (`transformer_4k`) and `1.9055` (`memory_4k`) to **1.8434** (`val_loss_nomem = 1.9184`, `-0.0750` nat causal memory gain) and raises gap-406 pet recall from `25.0%` chance to **37.5%**.
  - `cortex` (`CortexTransformer`): 5 `SynapseBlock`s + weight-tied input-anchored iterative workspace core (`ponder_steps=3`).
  - `hippocampus` (`HippocampusTransformer`, Gen-2 Wall-1 fix): 7 `SynapseBlock`s + 1 `HippocampusBlock` (last block only — "one hippocampus serving many cortical areas"; also 8x fewer slot kernels). The `HippocampusAttention` adds a 48-slot sparse landmark bank per head on top of the delta matrix: top-1 surprising token per micro-chunk writes (key,value,salience); modern-Hopfield softmax read at learned temperature (no distance decay); EMA merge at cos>0.92; least-salient eviction; EOT wipe. Layer state is a dict `{"M","K","V","w"}` — `forward_stream` handles mixed tensor/dict mems. 17.347M params. Train with `bash train.sh NAME --arch hippocampus`. At 24,000 steps (`runs/hippocampus_24k`, seed 0): **val_loss 1.5142 vs baseline 1.5888** (`val_loss_nomem=1.6158`, −0.1016-nat memory gain), gap-406 pet recall **60.4%** and gap-807 **42.5%** vs 25% chance (baseline ctx-capped to 256 → 25.0% at both). `hippocampus_w3` variant: `n_write_per_chunk=3` (top-3 surprising tokens/chunk; ~2.5x slower, run uncompiled: `.venv/bin/python lab/train.py NAME --arch hippocampus_w3`). **Exp #022 FALSIFIED** at 4k gate: w3 ≤ w1 at every recall gap (33.5% vs 36.7% @406 all-metric), val_loss bit-matched 1.8452 — write-*rate* is not the residual bottleneck (3× slot churn → more evictions); it's a write-*selection* problem. Config kept for future learned-allocation variants; `hippocampus` stays n_write=1.

## Multi-Axis Cognitive AGI Lab (`cortex/`, `cortex_runs/`, `research/`)

Modular 11-subsystem cognitive architecture (`SYNAPSE-1` / `H3_cortex_agi`) compared against controlled iso-parameter (~5.8M–6.2M, identical `W=48` working window) ablations (`H0_transformer`, `H1_synapse_mem`, `H2_latent_ponder`), plus Gen-2 unified `H4_cortex_v2` (5.950M):

- Gen-2 modules in `cortex/arch.py`: `HippocampalMemoryAttention` (delta matrix + 48-slot landmark bank per head, same write/merge/evict/reset rules as `lab` `hippocampus`), `DualMemoryBlockV2`, `SinusoidalLatentReasoner` (continuous sin/cos depth encoding × `step_gain` — stable ponder K=1..16+ — plus zero-init FiLM `program` conditioning for test-time program induction), `LatentWorldModel.milestone_head`/`progress_head` (subgoal machinery).
- Gen-2 planner in `cortex/planning.py`: `select_action_mpc(mode="subgoal")` stages beam search toward first-unachieved milestone (key->door->goal), quantized-latent `_dedup_beams`, 2-step latent unroll in `train_world_model` (`unroll_prob=0.25`), eval mode `hierarchical_subgoal_H8`. **Wall-2 solver:** `select_action_symbolic` — grounded symbolic beam search over the obs-parsed board (`obs[3:39]-COLOR_BASE` cells, `obs[2]-COLOR_BASE` milestone flags): exact state dedup (`agent_pos × flags`), terminal traps, BFS-to-subgoal shaping, milestone-consistent transition rules (`_symbolic_step`). Eval modes `symbolic_oracle` (obs-parsed board — 90% solve) and `symbolic_decoded` (learned `obs_dec` perception — 0%, perception gap). Re-grounded latent variant `select_action_mpc_reground` kept for the record (0% — imagination gap).

- Invariant tests (<5 s, CPU or GPU): `CUDA_VISIBLE_DEVICES="" .venv/bin/python cortex/test_invariants.py`
  (checks causality, multi-layer episode boundary isolation, fast-weight Frobenius norm bounds, TTT slow-weight reversibility, and `lab/` streaming equivalence).
- Full 8-domain benchmark suite (ID + OOD across Episodic Recall, $S_5$ State Tracking, Compositional Math, 2D ARC Grid Rule Induction + Test-Time Plasticity, Confounded Causal `do`-Calculus, Multimodal 2D Grounding, Key-Door-BaitTrap-Goal World-Model Beam MPC, and 5-Phase Continual Learning with dormant-neuron tracking):
  `.venv/bin/python cortex/run_suite.py --steps 3600 --batch 36 --seed 0`
- Post-grokking algebraic/causal stress test & $K=1..8$ compute scaling:
  `.venv/bin/python cortex/reasoning_grok.py`
- Autonomous diagnostic hypothesis, ablation & reversible evolutionary archive loop:
  `.venv/bin/python cortex/autonomous_loop.py --steps 3600 --batch 36 --seed 0 [--arch H4_cortex_v2]`
- Gen-2 verification scripts: `cortex/replicate_h4_seed1.py` (seed-1 causal-grok replication), `cortex/run_h4_continual.py` (standalone H4 5-phase continual eval → `cortex_runs/continual_H4_gen2.json`).
- **Gen-2 verified headline results** (`cortex_runs/benchmark_results_gen2.json`, `runs/hippocampus_24k`):
  `H4_cortex_v2` ID 0.3854 / OOD 0.3050 vs `H3` 0.2146/0.1671; `do()`-calculus grokked to ~1.0 within 3,600 steps on two seeds — **driven by the hippocampal slot bank as a training-time scaffold** (eval slot-kill −4pts, but `H3+sinusoidal` without slots never groks: 0.20–0.25 on two seeds; sinusoidal's role is K-extrapolation stability instead). **Wall-2 planning breached at the planning layer** (`symbolic_oracle` 90% solve / 0% trap on 40 held-out maps — the wall was state-abstraction fidelity, not search; learned-`obs_dec` perception variant still 0%). ARC exact-solve still 0% (FiLM+program TTT insufficient — needs discrete program search).
- Research knowledge base, literature database, architecture specification, benchmark reports & roadmap: `research/`.
