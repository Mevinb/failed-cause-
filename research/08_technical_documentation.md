# Technical Documentation & Reproducibility Manual (`agi-exi`)

**Status:** Living Engineering Documentation (Deliverable 8)  
**Workspace:** `/home/mevlec/Data/exp/agi-exi`  
**Hardware Profile:** Intel Core i5-12500H (16 threads, 15 GiB RAM), NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB VRAM, Compute Capability 8.9)

---

## 1. Repository Architecture Overview

The experimental codebase is organized into **three complementary research subsystems** sharing a unified Python 3.12 + PyTorch 2.11 (`cu130`) environment (`.venv/`):

```
/home/mevlec/Data/exp/agi-exi/
├── AGENTS.md                        # Laboratory operational & quick-reference guide
├── llama.cpp-exp/                   # Track A: Patched C++/CUDA llama.cpp worktree for Qwen3.5-9B dissection
├── harness/                         # Track A: Automated sweep runner, 42-probe QA/passkey suite & state surgery
│   ├── run.py                       # Perplexity + probe runner across env-var architectural hooks
│   ├── sweeps.py                    # Layer rewiring, depth looping, GDN decay, residual scaling & feedback
│   ├── state_exps.py                # Recurrent conv/SSM state transplant, zeroing, freezing & style steering
│   └── analyze.py                   # Compiles results/*.jsonl -> results/SUMMARY.md
├── lab/                             # Track B: Iso-parameter (17.3M) From-Scratch TinyStories LM & Memory Lab
│   ├── prepare.py                   # Byte-level BPE (8192 vocab) + uint16 memmap dataset builder
│   ├── model.py                     # Transformer, MemoryTransformer, SynapseTransformer, CortexTransformer
│   ├── train.py                     # Deterministic bf16 + torch.compile trainer with streaming memory eval
│   ├── recall.py                    # Counterbalanced 4-way forced-choice recall across 0..800 token gaps
│   └── sample.py                    # Story generation sampler
├── cortex/                          # Track C: Modular 11-Subsystem AGI Prototype (SYNAPSE-1) & 8-Domain Suite
│   ├── arch.py                      # H0_transformer, H1_synapse_mem, H2_latent_ponder, H3_cortex_agi (~5.3M)
│   ├── tasks.py                     # 6 core multi-task ID/OOD generators (Episodic, S5, Math, ARC, Causal, MM)
│   ├── planning.py                  # Key-Door-BaitTrap-Goal world-model JEPA trainer & parallel beam MPC
│   ├── continual.py                 # 5-phase continual learning & Dohare et al. plasticity preservation
│   ├── run_suite.py                 # Unified multi-hypothesis trainer, TTT, ECE/AUROC & compute-scaling eval
│   ├── autonomous_loop.py           # Darwin-Godel-inspired diagnostic hypothesis & reversible archive loop
│   └── test_invariants.py           # Automated causality, episode-isolation, norm-bound & TTT safety tests
└── research/                        # Scientific Deliverables 1-3, 6-10
```

---

## 2. Reproducing Every Experimental Track

### 2.1 Automated Mathematical & Safety Invariant Tests (CPU or GPU, <5s)
```sh
CUDA_VISIBLE_DEVICES="" .venv/bin/python cortex/test_invariants.py
```
Verifies:
1. **Strict Causality & Padding Invariance:** Corrupting future positions (`t > k`) with `<pad>` or `<reset>` alters past logits by $< 10^{-5}$.
2. **Multi-Layer Episode Boundary Isolation:** Prepending arbitrary prior episodes before `<reset>` alters post-reset predictions by $< 10^{-5}$ across stacked memory + attention layers.
3. **Fast-Weight Spectral/Frobenius Boundedness & TTT Reversibility:** Fast-weight associative matrices remain strictly bounded ($\|M\|_F < 50$) over 400-token streams, and Test-Time Training restores slow parameters bit-identically (`atol=1e-7`).
4. **Streaming Equivalence (`lab/model.py`):** Whole-sequence `forward(idx)` matches window-by-window `forward_stream(...)` within $< 10^{-5}$.

### 2.2 Track A — Pretrained Hybrid `Qwen3.5-9B` Dissection
Ensure no competing GPU processes are active (`nvidia-smi`), and confirm symlink `models/Qwen3.5-9B-Q4_K_M.gguf` points to the Q4_K_M GGUF weights:
```sh
.venv/bin/python harness/state_exps.py --only dissociate style recall
.venv/bin/python harness/run.py rewire --only drop_31 no_attn_mid attn_phase_shift
.venv/bin/python harness/run.py loops --only baseline loop2_12_15 loop2_12_15_damp rep2_L14 rep2_L15
.venv/bin/python harness/run.py decay --only baseline decay_0.25 decay_0.5 decay_2.0 decay_4.0
.venv/bin/python harness/run.py scale --only baseline scale_attn_0.0 scale_attn_0.5 scale_rec_0.5
.venv/bin/python harness/run.py feedback --only baseline fb_31_0_0.02 fb_31_0_0.1 fb_8_24_0.3
.venv/bin/python harness/analyze.py
```
Outputs append deterministically to `results/*.jsonl` and `results/SUMMARY.md`.

### 2.3 Track B — Iso-Parameter (`17.3M`) From-Scratch Language & Episodic Memory Lab
```sh
# Controlled 4,000-step iso-parameter comparison (~5-8 min per architecture)
bash train.sh transformer_4k --arch transformer --steps 4000 --eval-every 1000 --log-every 500
bash train.sh memory_4k      --arch memory      --steps 4000 --eval-every 1000 --log-every 500
bash train.sh synapse_4k     --arch synapse     --steps 4000 --eval-every 1000 --log-every 500
bash train.sh cortex_4k      --arch cortex      --steps 4000 --eval-every 1000 --log-every 500

# Score long-range recall across 0, 32, 96, 192, 400, 800 filler tokens
(cd lab && ../.venv/bin/python recall.py eval cortex_4k)
```

### 2.4 Track C — Multi-Axis Cognitive AGI Suite (`Cortex-AGI` / `SYNAPSE-1`)
```sh
# Full H0 vs H1 vs H2 vs H3 comparison + TTT + World-Model MPC + 5-Phase Continual Learning
.venv/bin/python cortex/run_suite.py --steps 3600 --batch 36 --seed 0 --out cortex_runs/benchmark_results_iter2.json

# Autonomous diagnostic hypothesis generation, ablation & reversible evolutionary loop
.venv/bin/python cortex/autonomous_loop.py --steps 3600 --batch 36 --seed 0
```

---

## 3. Hardware-Aware Memory & Optimization Practices

1. **`fp32` Fast-Weight State Under `bf16` Autocast:** Both `lab/model.py` and `cortex/arch.py` execute associative delta updates inside `torch.autocast(..., enabled=False)` so repeated multiplicative decay over hundreds of micro-chunks avoids `bfloat16` mantissa underflow.
2. **Branch-Free Vectorized Episode Reset:** Episode boundaries (`idx == 0`) are resolved via cumulative-sum masks (`ec.cumsum(1) == 0` and reverse cumsum) with zero host-device synchronization or `torch.compile` graph breaks.
3. **6 GB VRAM Guardrail:** Peak VRAM stays at `1.65–2.30 GiB` for 17.3M `lab/` runs and `< 0.75 GiB` for 5.3M `cortex/` runs, leaving $> 3.5\text{ GiB}$ headroom on the RTX 4050.
