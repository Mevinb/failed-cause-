# `agi-exi` — Master Documentation: Everything Built, Tested, Discovered & Learned

**Workspace:** `/home/mevlec/Data/exp/agi-exi`  
**Hardware:** Intel Core i5-12500H (16 threads, 15 GiB RAM) + NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB VRAM)  
**Last Updated:** 2026-11-13 (Living Master Document — updated after every experimental generation)

---

## Part 1: Plain-English Overview (What Is This Project & What Have We Done?)

This repository is a self-contained **experimental AI research laboratory** running entirely on a single 6 GB laptop GPU. Instead of just prompting existing chatbots or claiming hype, its goal is to **understand why today's AI architectures fall short of general intelligence, invent concrete architectural fixes, and test them head-to-head under fair, reproducible rules.**

Everything in this workspace is divided into **three connected tracks**:

### Track A — "Brain Surgery" on an Existing 9-Billion-Parameter AI (`llama.cpp-exp/` & `harness/`)
- **What we built:** A custom C++/CUDA fork of `llama.cpp` with surgical hooks (`EXP_LAYER_MAP`, `EXP_LAYER_SCALE`, `EXP_FEEDBACK`, `EXP_SSM_DECAY`) and a state-transplant tool (`llama-exp-state`) targeting **`Qwen3.5-9B`**, a modern hybrid model with **24 recurrent Gated-DeltaNet layers** (running mental summary) and **8 Full-Attention layers** (exact lookback memory).
- **What we discovered:**
  1. **Recurrent layers in hybrid LLMs don't store exact facts ("Attention-Shortcut Starvation"):** When we wiped **all 24** recurrent mental-summary layers to zero after the model read a secret code (`7391`) and name (`Zorbak`), the model **still recalled them 100%** using its 8 attention layers. Conversely, when we transplanted all 24 recurrent layers from a session that knew the secret code into a blank session, **0% of the facts transferred** (it guessed `1234`). And dropping just the final attention layer (`drop_31`) or three middle attention layers (`11, 15, 19`) crashed passkey memory from **100% to 0%**, while general knowledge and arithmetic stayed at **100%**.
  2. **Recurrent layers DO store language, mood, and grammar rhythm:** Blending the recurrent brain state from a **French** or **furious** paragraph into a calm English prompt about mild weather caused the model to answer in **fluent French** (*"Aujourd'hui, il fait beau..."*) or write about a **raging storm**. Freezing early/mid recurrent layers caused immediate word-stuttering loops (*"What is this is this..."*).
  3. **Layer 0 is uniquely irreplaceable:** Dropping middle recurrent layer 13 slightly *improved* perplexity (`8.376 -> 8.367`), whereas dropping Layer 0 destroyed the entire network (`PPL = 27,454`) because Layer 0 translates raw token embeddings into the shared coordinate space of all later layers.
  4. **You cannot just loop a normal model at test time to make it "think longer":** Forcing middle blocks (`12..15`) to repeat twice degraded perplexity (`8.376 -> 9.411`) because the model was never trained to iterate on its own thoughts.

### Track B — Training 17.3M-Parameter Story Models From Scratch (`lab/` & `runs/`)
- **Why we built it:** Track A showed that long-term memory stays lazy if local attention can see the whole training text. So we built a clean from-scratch training lab on **TinyStories** (458M tokens, 2.1M stories, median story length **183 tokens**) where every model has the **exact same size (`17.3M` parameters)**, exact same data, and exact same training steps.
- **What we tested & achieved:**
  1. **`baseline` / `transformer_4k` (Standard Transformer):** Sees only the last 256 tokens. Great inside 256 tokens, **100% blind (`25.0%` random guessing)** on facts 400 or 800 tokens back.
  2. **`memory_4k` (Naive Chunk Memory Transformer):** Updates a memory matrix once every 256 tokens without clearing memory when one story ends (`EOT` token `0`) and a new unrelated story begins. Because 84.7% of TinyStories are shorter than 256 tokens, unrelated stories bled together: validation loss got slightly *worse* (`1.9055` vs `1.9029`) and 400-token recall stayed near chance (`26.9%`).
  3. **`synapse_4k` & `synapse_v2_4k` (Episode-Aware, Surprise-Gated Memory):**
     - Shrank local attention to **64 tokens** (forcing the model to use long-term memory on every story >64 tokens during training).
     - Updated memory every **32 tokens** using a decoupled, position-free Delta rule that writes strongly on **surprising** tokens and **wipes clean on `EOT` (`id=0`)** between stories.
     - After diagnosing in Cycle 1 (`synapse_4k`, `val_loss = 1.8505`) that decaying by `0.95` eight times per window ($0.95^8 = 0.68$) forgot too fast, Cycle 2 (`synapse_v2_4k`) slowed per-chunk decay (`0.996`), sparsified filler writes, and normalized value vectors—reaching **`val_loss = 1.8434`** (`5.8%` better perplexity than baseline; `-0.0750` nats coming directly from memory!) and lifting 406-token pet-name recall from `25.0%` blind guessing to **`37.5%`**.

### Track C — Building a Multi-Subsystem Cognitive Prototype (`cortex/` & `cortex_runs/`)
Language prediction alone doesn't test reasoning, cause-and-effect, planning, or lifelong learning. We built **`Cortex-AGI` (`SYNAPSE-1`, ~5.8M parameters)** and tested it head-to-head against equal-sized Transformer (`H0`), Memory-Only (`H1`), and Ponder-Only (`H2`) baselines across **8 cognitive benchmarks**:
1. **Long-Horizon Memory (`Episodic Recall`):** Remembered key-value pairs at **6.7x** its 48-token working window (`2.5x–3.4x` chance accuracy), where standard Transformers hit pure chance (`5.8%`). Turning off the "surprise gate" (`ablation_no_surprise_gate`) caused an **11% relative drop** in out-of-distribution accuracy, proving surprise-driven memory works.
2. **Thinking Longer on Harder Problems (`Iterative Latent Reasoner`):** Weight-shared recurrent layers loop over an internal workspace anchored to the input. Once symbols were grounded, giving the model more internal thinking loops (`K=1 -> 2 -> 3 -> 4 -> 6`) steadily improved out-of-distribution object-shuffling ($S_5$) accuracy (`22.0% -> 23.5%`), breaking below the loss floor where fixed-depth models got stuck (`1.3158` vs `1.4405`).
3. **Cause vs. Coincidence (`Causal do-Calculus`):** Past the grokking threshold (~1,800 steps), achieved **100% accuracy** on unseen causal systems when asked *"what happens if I force X to change?"* (`do(X=1)`), where relying on correlation scores `0%`.
4. **Learning New Rules at Test Time (`Test-Time Training / TTT`):** Given just 3 demo pairs of a brand-new 2D visual grid rule, adapting a tiny 16-rank scratchpad adapter boosted cell accuracy (`+3.1%`) while keeping 100% of permanent weights untouched.
5. **Imagining Before Acting (`Latent World-Model MPC Planning`):** In $6 \times 6$ `Key-Door-BaitTrap-Goal` mazes with a tempting `+0.5` dead-end trap right next to the start, greedy 1-step agents fell into the trap **61.3%** of the time; simulating 6 steps ahead in our learned world model cut trap deaths to **7.5%** (**8.2x safer**).
6. **Lifelong Learning Without Neuron Death (`Continual Plasticity Engine`):** Training sequentially on 5 different tasks caused normal AdamW to forget old tasks (`BWT = -0.161`) and **kill 32.9% of its SwiGLU neurons**. Replaying high-surprise memories + gentle weight refresh ("shrink-and-perturb") kept dead neurons down to **0.36%–2.04%** (**16x–57x healthier**) and achieved **5x higher** final multi-task accuracy.
7. **Self-Checking Safety Loop (`autonomous_loop.py`):** Our automated evolutionary loop caught a real hidden bug in Iteration 2 where un-normalized memory values caused matrix norms to swell (`69.8 > 50.0` safety limit) and blocked promotion until sphere-normalization fixed it (`norm = 11.35`, boosting validation OOD accuracy by **`+72.7%`**).

---

## Part 2: Complete Directory & File Map

```
/home/mevlec/Data/exp/agi-exi/
├── DOCUMENTATION.md                       # THIS FILE: Complete plain-English + technical guide
├── AGENTS.md                              # Concise operational rules & CLI cheat-sheet
├── train.sh / train_memory.sh             # Resumable TinyStories training launchers (VRAM + sleep guards)
│
├── llama.cpp-exp/                         # Patched C++/CUDA llama.cpp worktree (branch agi-exp, base bebc935)
│   ├── src/llama-exp.{h,cpp}              # Env-var parser for LAYER_MAP, LAYER_SCALE, FEEDBACK, SSM_DECAY
│   ├── src/models/qwen35.cpp              # Hooked Qwen3.5 compute graph (Gated-DeltaNet + Full Attention)
│   └── tools/exp-state/                   # C++ tool for recurrent conv/SSM state transplant/blend/zero/freeze
│
├── harness/                               # Automated Qwen3.5-9B experimental sweep & probe framework
│   ├── sweeps.py                          # Definitions of rewire, loops, decay, scale, feedback sweeps
│   ├── probes.py                          # 42-item deterministic QA (fact, arith, seq, lang) + 6 passkey tests
│   ├── run.py                             # Runs llama-perplexity + llama-server probe battery -> results/*.jsonl
│   ├── state_exps.py                      # Runs recall, dissociation & style-transfer state surgery -> state.jsonl
│   └── analyze.py                         # Generates results/SUMMARY.md from all results/*.jsonl
│
├── results/                               # Verified Qwen3.5-9B outputs
│   ├── SUMMARY.md                         # Master Markdown tables of all Qwen3.5 sweeps & state surgery
│   └── {baseline,rewire,loops,decay,scale,feedback,state}.jsonl
│
├── lab/                                   # Iso-parameter (17.3M) From-Scratch TinyStories LM Lab
│   ├── prepare.py                         # Trains 8192-vocab byte BPE; encodes train.bin (458M) & val.bin (4.6M)
│   ├── model.py                           # Transformer, MemoryTransformer, SynapseTransformer, CortexTransformer (+Gen-2)
│   ├── train.py                           # Deterministic bf16 + torch.compile trainer & streaming evaluator
│   ├── recall.py                          # 2,880-item counterbalanced 4-way recall benchmark (0..800 token gaps)
│   └── sample.py                          # Text generator from any runs/<NAME>/last.pt checkpoint
│
├── runs/                                  # Saved TinyStories checkpoints, training logs & recall scores
│   ├── baseline/                          # 24,000-step Transformer (val_loss 1.5888, recall 25.0% @ >=400)
│   ├── transformer_4k/                    # 4,000-step Transformer control (val_loss 1.9029, recall 25.0% @ >=400)
│   ├── memory_4k/                         # 4,000-step chunk MemoryTransformer (val_loss 1.9055, recall 26.9% @ 406)
│   ├── synapse_4k/                        # 4,000-step Synapse Cycle-1 (val_loss 1.8505, recall 31.2% / 34.6% pet @ 406)
│   ├── cortex_4k/                         # 4,000-step Cortex LM Cycle-1 (val_loss 1.8529)
│   └── synapse_v2_4k/                     # 4,000-step Synapse Cycle-2 (val_loss 1.8434, recall 33.5% / 37.5% pet @ 406)
│
├── cortex/                                # Modular Multi-Subsystem Cognitive AGI Testbed
│   ├── arch.py                            # 11-subsystem modules + H0, H1, H2, H3 (& Gen-2 H4) architectures
│   ├── tasks.py                           # 6 ID+OOD generators (Episodic, S5, Math, 2D ARC, Causal do, Multimodal)
│   ├── planning.py                        # Procedural 6x6 Key-Door-BaitTrap-Goal env, JEPA WM & Lookahead Planner
│   ├── continual.py                       # 5-phase sequential learning, Surprise Replay & Shrink-and-Perturb
│   ├── run_suite.py                       # Master trainer, TTT evaluator, ECE/AUROC & compute-scaling suite
│   ├── reasoning_grok.py                  # Post-grokking 540k-episode algebraic & causal stress test
│   ├── autonomous_loop.py                 # Diagnostic hypothesis generator, safety gate & evolutionary archive
│   └── test_invariants.py                 # Automated causality, episode-isolation, norm-bound & TTT unit tests
│
├── cortex_runs/                           # All saved Cortex-AGI checkpoints & JSON logs across iterations
│   ├── benchmark_results_iter{1,2,3}.json # Full per-axis metrics across Iterative Engineering Cycles 1-3
│   ├── evolution_archive*.jsonl           # Complete log of accepted & rejected self-improvement mutations
│   ├── reasoning_grok_results.json        # Post-grokking S5 + Math + Causal do-calculus results
│   └── s5_grok_comparison.json            # Symbol-grounded S5 OOD & K=1..8 latent compute scaling
│
└── research/                              # Scientific Knowledge Base, Literature DB & Reports
    ├── MASTER_EXPERIMENT_LOG.md           # Chronological lab notebook of every experiment (#001+)
    ├── 01_knowledge_base.md               # First principles & Top-10 unresolved AGI bottlenecks
    ├── 02_literature_database.json        # Structured primary-source database (15 foundational papers)
    ├── 02_literature_review.md            # Synthesis distinguishing proven theorems from speculation
    ├── 03_architecture_spec.md            # Mathematical specification of H0, H1, H2, H3 (SYNAPSE-1)
    ├── 06_benchmark_report.md             # Full quantitative tables across Tracks A, B, and C
    ├── 07_hypotheses_and_open_problems.md # Falsifiable hypothesis registry & verdicts
    ├── 08_technical_documentation.md      # Reproducibility commands & hardware optimization guide
    ├── 09_roadmap.md                      # Short-, medium-, and long-term AGI engineering roadmap
    └── 10_progress_reports.md             # Cycle-by-cycle progress & failure retrospectives
```

---

## Part 3: Core Mathematical Mechanisms Implemented So Far

### 3.1 Bounded, Episode-Aware, Surprise-Gated Delta Fast-Weight Memory (`Synapse` & `Cortex-AGI v1`)
For each head $h$ with decoupled, RoPE-free normalized query/key $q_t, k_t \in \mathbb{S}^{d_m - 1}$ and sphere-normalized value $v_t = \sqrt{d_m}\,\tilde{v}_t / \|\tilde{v}_t\|_2$:
1. **Episode Isolation:** On episode start token (`id = 0`), cumulative episode masks prevent both local sliding-window attention and fast-weight reads/writes from leaking across episode boundaries (`diff < 10^-6` verified by `cortex/test_invariants.py`).
2. **Surprise-Modulated Delta Step:** Within micro-chunk $c$ of size $M_c$:
   $$e_t = v_t - k_t M_{c-1}^{\text{dec}}, \qquad s_t = \frac{\|\text{sg}(e_t)\|_2}{\sqrt{d_m}}, \qquad \beta_t = \sigma(b_t + \gamma_s s_t)$$
   $$M_c = M_{c-1}^{\text{dec}} + \frac{K_c^\top (\beta_c \odot E_c)}{1 + \sum_{t \in c} \beta_t}$$
   Because $\|k_t\|_2 = 1$ and $\|v_t\|_2 = \sqrt{d_m}$, the Frobenius norm is **provably bounded** ($\|M_c\|_F \le d_m$) over arbitrarily long sequences.

### 3.2 Remaining Bottlenecks Targeted in Generation 2 (Active Next Steps)
1. **Wall 1 — Linear Matrix Interference ($O(d_m)$ rank bound):** Over 800+ filler tokens, summing dozens of rank-1 updates into a $32 \times 32$ matrix blurs early memories (`30.0%` recall at gap 807). **Gen-2 Solution:** Add a **Hippocampal Sparse Landmark Slot Bank** (Modern Hopfield high-inverse-temperature retrieval over discrete competitive surprise-gated key-value slots—zero filler interference over arbitrary distances).
2. **Wall 2 — Flat 6-Step Planning Horizon (`0%` full 14-step `Key->Door->Goal` solve):** 6-step primitive action beams dodge adjacent traps (`92.5%`), but cannot see 14 steps ahead. **Gen-2 Solution:** Multi-step unrolled JEPA training + **State-Deduplicated Latent Best-First Tree Search** guided by learned subgoal milestones (`has_key`, `door_open`, `at_goal`).
3. **Wall 3 — Out-of-Support Recurrence Drift (`K > K_train`) & Unseen 2-Op ARC Exact Grid Solves:** Learned discrete step embeddings `step_emb[k]` are untrained for $k \ge K_{\text{train}}$, causing accuracy to dip at $K=8$. **Gen-2 Solution:** Continuous sinusoidal/contractive depth conditioning with progressive-detach fixed-point training (`K=1..16` stability) + **Latent Task-Program FiLM Conditioning** refined at test time for exact full-grid compositional rule induction.

---

## Part 4: Generation-2 Upgrades & Breakthroughs

### 4.1 The Three Mechanisms Built to Break the Walls

**Wall 1 — `HippocampalSlotMemory` (discrete landmark slots that never decay)**
*Files:* `lab/model.py` (`HippocampusAttention`, `HippocampusBlock`, `HippocampusTransformer` = `hippocampus`), `cortex/arch.py` (`HippocampalMemoryAttention`, `DualMemoryBlockV2`).

*How it works (plain words):* the Delta matrix is a smooth mental summary — great for recent context but it blurs facts written 800 tokens ago. The slot bank is a little notebook of **48 discrete "landmark" pages** per head: every 32 tokens, the single most surprising token writes its key and value on a fresh page (or merges into a nearly-identical page, or overwrites the least-important page when full). Reading is a **content lookup**, not a memory-of-a-memory — so a fact on page 3 is just as sharp whether it was written 8 or 800 tokens ago. Like the matrix, the whole notebook is wiped clean when a new story starts (`EOT` token `0`).

*Why it's different from attention:* softmax attention over past tokens costs O(T) memory and (in our W=64 design) isn't available beyond the local window; slots are O(48) fixed storage that persist arbitrarily far. *Why it's different from the matrix:* the matrix superimposes ~rank-limited updates that interfere; slots store events discretely and competitively.

*Verified at 4k budget (Exp #014):* `hippocampus_4k` `val_loss = 1.8452` vs `synapse_v2_4k 1.8434` (slots cost nothing on LM learning) while lifting long-gap recall: gap-400 pet **43.8%** vs 37.5% (chance 25%), gap-800 pet **36.7%** vs 32.9%. First lab architecture whose recall does not fully collapse to chance at 800 tokens. *Known trade-off:* gap-0 recall dips (57.9% vs 71.7%) — the slot-read gate (open ~38% at init) injects retrieval noise on queries local attention could answer; expected to improve as training teaches the gate when to stay shut.

**Wall 2 — Hierarchical Subgoal World-Model Planner**
*Files:* `cortex/arch.py` (`LatentWorldModel.milestone_head`, `.progress_head`), `cortex/planning.py` (`mode="subgoal"`, `_dedup_beams`, 2-step unroll in `train_world_model`, eval mode `hierarchical_subgoal_H8`).

*How it works:* instead of one flat 6-step search toward the distant goal, the planner first asks the world model *"which milestone is already achieved?"* (`milestone_head(z) → P[has_key, door_open, done]`), then beam-searches toward the **next unachieved milestone only** (`progress_head(z,a) → P[next step gets key / opens door / reaches goal]`), with latent-state deduplication so the beam doesn't waste slots re-imagining the same board, and 2-step unrolled JEPA training so imagined rollouts stay accurate deeper. Re-planning every real step turns a 14-step problem into three ~5-step problems.

**Wall 3 — Sinusoidal Contractive Reasoner + FiLM Program Conditioning**
*Files:* `cortex/arch.py` (`sinusoidal_step_encoding`, `SinusoidalLatentReasoner`), `cortex/run_suite.py` (program-induction in `evaluate_arc_ttt`).

*How it works:* Gen-1's "think longer" loop used a learned embedding per step index — embedding #7 and #8 were never trained, so unrolling past K=6 degraded. Gen-2 encodes depth as a continuous sin/cos signal × a learned gain — mathematically defined for ANY K, so K=12 or K=16 stays on-manifold. Added FiLM conditioning: a `program` vector (zero → exact identity) scales and shifts every refinement step — a slot where a *task program* can be written at test time by gradient (the TTT loop now optimizes it alongside the rank-16 adapter on H4).

**`H4_cortex_v2` (5.950M params):** `DualMemoryBlockV2`×3 (delta matrix + slots in every memory block) + `SinusoidalLatentReasoner` + subgoal world-model heads. Registered in `ARCH_REGISTRY`; `run_suite.py` evaluates it on all axes including ponder scaling to K=16 and subgoal planning.

### 4.2 Generation-2 Verified Results (what each wall-breaker actually delivered)

**Wall 1 — BREACHED (flagship scale).** `hippocampus_24k` vs 24k `baseline` transformer: `val_loss` **1.5142 vs 1.5888** (better LM *and* better memory; `val_loss_nomem = 1.6158` → **−0.1016 nats** causal memory gain, up from −0.0756 at 4k). Pet-name recall: gap-406 **60.4%** vs baseline 25.0% = **2.4× chance**; gap-807 **42.5%** = **1.7× chance** — first model in the lab holding >40% recall across 800+ tokens of filler. The 4k gap-0 dip healed with training (57.9%→91.2%): the slot gate learned to close on easy queries. Residual cost only at gap≤103. Artifact: `runs/hippocampus_24k/`.

**Wall 2 — NOT breached (honest negative).** Hierarchical subgoal planning underperformed flat MPC (trap 51.3% vs 22.5%); full 14-step Key→Door→Goal stays **0%** for every planner. Partially-trained milestone heads mislabel the stage → actively mislead search. Next levers: symbolic subgoal state, MCTS/best-first with env-state decoding, longer WM training.

**Wall 3 — partially breached.** Sinusoidal depth is stable to K=16 (reasoning-OOD 0.232 vs K=6 peak 0.251 — graceful decay, not the Gen-1 crash into untrained embeddings). Exact full-grid ARC program induction still **0%** — FiLM+program TTT lifts cell accuracy (+1.5 pts) but not exact solves; needs discrete program search, not just gradient refinement.

**`H4_cortex_v2` benchmark:** ID 0.3854 / OOD 0.3050 vs H3 0.2146/0.1671 (+83% rel.). The headline is a **verified, replicated causal-reasoning breakthrough**: `do()` interventional inference grokked to ~1.0 within 3,600 steps on *two independent seeds* (seed-0: 0.98, seed-1: 1.00) — H3 needed ~540k episodes (~150× more data). Mechanism attribution (Exp #015b/c): slot-kill costs only ~4 pts, FiLM is inert at eval, lottery ruled out → **the sinusoidal depth encoding is the breakthrough mechanism** (smooth depth-sharing trains all ponder depths every step vs ~1/5 per discrete embedding). The slot bank's *verified* contribution is episodic recall, not causal reasoning — an important correction to the initial read.

### 4.3 Running / Next
- `H4` 5-phase continual standalone eval → `cortex/run_h4_continual.py` (running)
- Next memory levers: per-head write-top-k, learned write allocation, second slot bank at mid-depth (gap≤103 residual cost), multi-fact saturation study.
- Next reasoner lever: `H3+sinusoidal` (no slots) ablation to isolate depth-sharing as the causal-grok driver; adaptive halting on the `dz` signal (ponder curve saturates at K=6 — spend compute only where it helps).

*(See `research/MASTER_EXPERIMENT_LOG.md` Exp #013–#016 for the chronological record.)*
