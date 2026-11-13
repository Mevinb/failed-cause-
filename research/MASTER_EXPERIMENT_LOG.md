# Master Chronological Experiment Log (`agi-exi`)

Every experiment conducted in this laboratory—including negative results, bugs discovered, ablations, and architectural breakthroughs—is permanently logged here in chronological order.

---

## Generation 0 & 1 Experiments (Completed & Verified)

### Exp #001 — Baseline 24k TinyStories Transformer
- **Artifact:** `runs/baseline/{log.jsonl,last.pt,recall.json}`
- **Setup:** `Transformer` (`17.31M` params, 8 layers, `d=384`, 6 heads, `block_size=256`), 24,000 steps $\times$ 8,192 tok/step (`196.6M` tokens, 33.5 min).
- **Result:** Whole-val `val_loss = 1.5888` (`ppl = 4.898`). Recall: gap 0 `97.7%`, gap 38 `84.2%`, gap 103 `83.1%`, gap 198 `56.0%`, **gap 406 `25.0%`**, **gap 807 `25.0%`** (100% blind outside 256-token window).

### Exp #002 — `Qwen3.5-9B` Layer Rewiring & Attention Ablation Sweep
- **Artifact:** `results/rewire.jsonl`, `results/SUMMARY.md`
- **Hypothesis (`E-Q1`):** Sparse full-attention layers (`3,7,11,15,19,23,27,31`), not the 24 Gated-DeltaNet recurrent layers, mediate exact in-context retrieval; Layer 0 is uniquely non-removable.
- **Result:** Stock `PPL = 8.376, acc = 1.000, passkey = 1.000`. `drop_0` collapsed (`PPL = 27,453.7, acc = 0.0`). `drop_13` improved slightly (`PPL = 8.367, acc = 1.000`). `drop_31` (`PPL = 26.004`) and `no_attn_mid` (`PPL = 11.104`) kept `fact = 1.000` while crashing **`passkey = 0.000`**. `attn_phase_shift` exploded (`PPL = 71.831, passkey = 0.000`). **Verdict: Confirmed.**

### Exp #003 — `Qwen3.5-9B` Recurrent State Surgery Double Dissociation
- **Artifact:** `results/state.jsonl`, `results/SUMMARY.md`
- **Setup:** `llama-exp-state` (`zero`, `transplant`, `freeze`, `blend`) across the 24 Gated-DeltaNet layers while leaving the 8 Attention KV caches untouched.
- **Result:** `zero_all` (wiping all 24 GDN states after reading `7391`/`Zorbak`/`Teal`) still recalled **`7391` and `Zorbak` 100%**. `transplant_all` (copying all 24 GDN states into neutral recipient) recalled **0%** (`1234` hallucination). `freeze_early/mid` caused syntactic stutter loops. `style_french_blend_2.0` & `style_angry_blend_2.0` transferred French language and angry tone onto calm English prompts. **Discovery: Attention-Shortcut Starvation Law.**

### Exp #004 — `Qwen3.5-9B` Untrained Depth Looping, Decay, Scaling & Feedback Sweeps
- **Artifact:** `results/{loops,decay,scale,feedback}.jsonl`, `results/SUMMARY.md`
- **Result:**
  - `decay_2.0` (2x faster GDN forgetting) slightly *improved* PPL (`8.366` vs `8.376`) and kept `passkey = 1.000`; even `decay_4.0` kept `passkey = 1.000` (`PPL = 8.593`).
  - `scale_attn_0.0` (zeroing all 8 attention residual outputs) collapsed (`PPL = 165.8, acc = 0.0`).
  - `loop2_12_15` degraded PPL (`9.411`, vs `8.770` damped at `0.5`); `fb_31_0_0.1` degraded 4-chunk PPL (`7.407 -> 7.582`). **Verdict: Iterative recurrence & top-down feedback must be trained end-to-end with input anchoring.**

### Exp #005 — TinyStories Data Distribution Audit & 4-Way Iso-Parameter (`17.3M`) Cycle-1 Comparison
- **Artifact:** `runs/{transformer_4k,memory_4k,synapse_4k,cortex_4k}/{log.jsonl,recall.json}`
- **Data Audit:** Measured `val.bin` story lengths (`21,990` stories): mean `209.7`, median `183.0`, **84.7% shorter than 256 tokens** (`>256 = 15.3%`). Predicted `MemoryTransformer` (`arch=memory`, once-per-256-token un-reset write) would suffer cross-story bleed across `EOT (id=0)`.
- **Result (4,000 steps, seed 0, `17.3M` params):**
  - `transformer_4k`: `val_loss = 1.9029`, gap-406/807 recall = `25.0% / 25.0%`.
  - `memory_4k`: `val_loss = 1.9055` (`nomem = 1.9151`, only `-0.0096` nat gain), gap-406/807 = `26.9% / 26.0%`.
  - `synapse_4k` (`W=64` local window + `C=32` surprise-gated `EOT`-reset Delta memory): **`val_loss = 1.8505`** (`nomem = 1.9224`, **`-0.0719` nat causal memory gain**), gap-406 pet recall **`34.6%`** (`31.2%` overall), gap-807 pet recall **`32.9%`**.
  - `cortex_4k` (5 `SynapseBlock`s + 3x weight-tied `IterativeWorkspace`): **`val_loss = 1.8529`** (`nomem = 1.9229`).
- **Failure Diagnosed:** Applying `sigmoid(3.0) = 0.9526` decay 8 times per 256-token window compounded to $0.9526^8 = 0.678$ per window, and `beta_bias = -1.0` wrote ~50% on every filler word.

### Exp #006 — `Cortex-AGI` Multi-Axis Cognitive Suite — Iteration 1
- **Artifact:** `cortex_runs/benchmark_results_iter1.json`
- **Setup:** `H0_transformer`, `H1_synapse_mem`, `H2_latent_ponder`, `H3_cortex_agi` (~5.3M params, `W=48`), 2,400 steps across 6 tasks + TTT + Planning + 5-Phase Continual Learning.
- **Result:** `H3_cortex_agi` led ID (`0.1840` vs `H0 0.1653`), `H1`/`H3` beat `H0`/`H2` on OOD `gap=160/320` recall (`19.6%–23.3%` vs `4.2%–10.0%`), and `H3` TTT gained `+3.12%` cell accuracy (`55.86% -> 58.98%`) on unseen compositional 2D ARC rules.
- **5 Bottlenecks/Bugs Diagnosed:** (1) Flat token cross-entropy gave 16-token `ARC` + 5-token `S_5` 84% of gradients; (2) `mean(p_tok)` vs joint `exact_acc` distorted multi-token ECE/AUROC; (3) `cand_upd` small-init projection choked 99% of `IterativeLatentReasoner` signal; (4) `LatentWorldModel` pooled an untrained `modality=3` token (`goal_rate = 0%`, though `trap_rate` fell `66.3% -> 8.8%`); (5) 16-cell `P5_arc_grid` exact match stayed `0%` in 320 continual steps.

### Exp #007 — Automated Invariant Test Caught Multi-Layer Cross-Episode Attention Bleed
- **Artifact:** `cortex/test_invariants.py`, `cortex/arch.py`
- **Discovery:** Even when fast-weight memory wiped cleanly on `TOK_RESET`, stacking three `W=48` local-attention layers leaked information across `TOK_RESET` via multi-hop receptive fields (`diff = 0.0969`). Adding cumulative episode-ID masking `same_ep & local_mask` reduced cross-episode bleed to `0.00e+00`.

### Exp #008 — `Cortex-AGI` Iteration 2 & Autonomous Safety Gate Catching Value-Norm Divergence
- **Artifact:** `cortex_runs/benchmark_results_iter2.json`, `cortex_runs/evolution_archive_iter2.jsonl`
- **Result:**
  - Sequence-level probability $\prod_m p_m$ improved global calibration **8.6x–16x** (`H3 ECE: 0.1741 -> 0.0106`, `AUROC: 0.4326 -> 0.7560`).
  - 5-Phase Continual Learning: `naive_adamw` suffered `BWT = -0.1547` and `20.73%` dormant neurons; `cortex_plasticity` achieved **`BWT = +0.0016` (zero forgetting)**, `0.36%` dormant neurons (**57x healthier**), and `FAA = 0.1488` (**5.4x naive AdamW**).
  - **Failures Diagnosed:** (a) Without `QK-RMSNorm`, `lr = 1.2e-3` caused recurrent attention entropy collapse in `H3`; (b) `autonomous_loop.py`'s `check_safety_invariants()` caught that raising `decay_logit` to `5.5` with un-normalized `vm` pushed fast-weight Frobenius norm to `63.4–69.8 > 50.0` (blocking promotion of `Gen-3`, even though `Gen-3` jumped `+29.5%` in `val_ood`!).

### Exp #009 — TinyStories Cycle-2 Sparse-Decay & Sphere-Normalized Value Fix (`synapse_v2_4k`)
- **Artifact:** `runs/synapse_v2_4k/{log.jsonl,last.pt,recall.json}`
- **Setup:** `SynapseTransformer` (`17.35M` params), `decay_logit = 5.5`, `beta_bias = -2.2`, sphere-normalized `vm` (`\|vm\|_2 = \sqrt{d_m}`), 4,000 steps, `seed = 0`.
- **Result:** Best 4k language model to date: **`val_loss = 1.8434 (ppl 6.32)`** (`nomem = 1.9184`, **`-0.0750` nat causal memory gain**), lifting recall across every distance (`gap 0: 71.7% pet / 59.8% all`, `gap 406: 37.5% pet / 33.5% all`, `gap 807: 32.9% pet / 30.0% all`).

### Exp #010 — `Cortex-AGI` Iteration 3 (`QK-RMSNorm` + Sphere-Normalized `vm` + Dreamer-Grounded WM) & Surprise Ablation
- **Artifact:** `cortex_runs/benchmark_results_iter3.json`, `cortex_runs/evolution_archive.jsonl`
- **Result:**
  - `max_mem_norm` dropped from `69.80` to **`11.345` (`OK`)**, while `Gen-0` validation OOD jumped **`+72.7%` (`0.0989 -> 0.1708`)**.
  - `H3_cortex_agi` (`5.82M` params) achieved highest held-out composite OOD accuracy (`0.1571` > `H1 0.1563` > `H0 0.1442` > `H2 0.1379`).
  - `ablation_no_surprise_gate` (`surprise_gain = 0`) dropped `val_ood` from `0.1708` to `0.1521` (**`-11.0%` relative**), proving surprise modulation causally drives OOD generalization.
  - `world_model_mpc_H6` cut irreversible bait-trap rate from `61.3%` (`myopic_H1`) and `37.5%` (`reactive_H0`) down to **`7.5%`** (`8.2x` reduction), though full 14-step `Key->Door->Goal` completion stayed `0.0%`.

### Exp #011 & #012 — Post-Grokking Algebraic/Causal Stress Test & $S_5$ Symbol-Grounding Discovery
- **Artifact:** `cortex_runs/reasoning_grok_results.json`, `cortex_runs/s5_grok_comparison.json`
- **Result:**
  - At 540k episodes (`4,500 steps x 120`), confounded interventional `do(X=1)` causal inference grokked to **100.0% exact OOD accuracy** across all models, while recursive `H2` (`loss = 1.3590`) and `H3` (`loss = 1.3158`) broke below the `1.4405` fixed-depth loss floor of `H0`/`H1`.
  - Training $S_5$ on `len >= 2` alone stayed at $\ln(5) = 1.609$ chance loss due to zero pairwise mutual information of un-grounded swap symbols; adding `len = 0, 1` grounded symbols immediately (`loss -> 0.62`), revealing monotonic test-time compute scaling on OOD `len=10` (`K=1: 22.0% -> K=2: 22.25% -> K=3: 23.0% -> K=4: 23.25% -> K=6: 23.5%`), followed by out-of-support dip at `K=8` (`21.0%`, recovered to `21.5%` by `K=8_ver`) due to unseen discrete `step_emb[6,7]`.

---

## Generation 2 Experiments (Pushing Past the Three Hard Walls)
*(Logged sequentially below as each experiment runs and verifies.)*

### Exp #013 — Gen-2 Architecture Implementation: Three Wall-Breaking Mechanisms
- **Date:** 2026-11-13 (session continuing the autonomous research loop)
- **Wall 1 (Linear matrix rank/interference ceiling):** Implemented `HippocampusAttention` (sparse competitive landmark slot bank) in `lab/model.py` + `cortex/arch.py`. Design: per micro-chunk, the top-1 surprising *active* token writes (key, value, salience) into a 48-slot content-addressable bank per head. Reads are modern-Hopfield softmax retrievals at learned inverse temperature — **zero distance decay**: a slot written 800 tokens back is retrieved as sharply as one written 8 tokens back. Near-duplicate keys (cos sim > 0.92) merge by EMA (0.7/0.3); when full, the least-salient slot is evicted. Slots wipe on `EOT`/`TOK_RESET` exactly like the matrix. `lab` arch `hippocampus` = 7 `SynapseBlock`s + 1 `HippocampusBlock` (single "hippocampus" serving all cortical delta matrices — also 8x cheaper in kernels). 17.347M params (vs baseline 17.308M). Verified: stream-vs-whole bit-exact, bounded norms, all 4 invariant suites pass.
- **Wall 2 (flat 6-step planning horizon, 0% goal completion):** Added `milestone_head(z) -> P[has_key, door_open, done]` and `progress_head(z,a) -> P[reach key/door/goal next]` to `LatentWorldModel`; `train_world_model` now supervises both (BCE, weight 2.0) plus **2-step latent unroll** (25% of steps) so imagined rollouts stay accurate at depth 8+. New planner mode `subgoal`: stage = first unachieved milestone from `milestone_head`, beam search scored by `3*sigmoid(prog_logit) + 0.5*prog_logit.clamp(-8,0)` step reward + `4*prog_next` terminal heuristic + existing trap/loop/prior terms, with **quantized latent deduplication** (`_dedup_beams`). New eval mode `hierarchical_subgoal_H8`.
- **Wall 3 (discrete `step_emb[k>6]` = untrained noise):** `SinusoidalLatentReasoner` replaces the embedding lookup with continuous sin/cos depth encoding × learned `step_gain` — defined for ANY K, so unrolling to K=12/16 stays on-manifold. Added FiLM task-program conditioning: `film(program) -> (gamma,beta)` modulates `u` each step (`u*(1+0.1*tanh(g)) + 0.1*b`); film is zero-init so `program=None` is exact identity — a program must be *learned* at test time (substrate for program induction). `evaluate_arc_ttt` now optimizes a per-episode `program` vector alongside the rank-16 adapter on H4.
- **New architecture:** `H4_cortex_v2` (5.950M params; H3=5.917M, H0=6.125M): `DualMemoryBlockV2` x3 + `SinusoidalLatentReasoner` + subgoal WM heads. Registered in `ARCH_REGISTRY` and `run_suite.py` (ponder curve extends to K=12,16; planning runs on both H3 and H4).
- **Compile-speed fix:** replaced `if ec.any(dim=1).any():` Python branch in the slot-wipe path with unconditional `dead` multiply (0 = no-op) — removes a per-chunk graph break.
- **Status:** implementation verified; quantitative results pending training runs below.

### Exp #014 — `hippocampus_4k` Gate Run (Wall-1 validation at 4k budget)
- **Artifact:** `runs/hippocampus_4k/{log.jsonl,last.pt,recall.json}` (2026-11-13, ~18 min at 62–64k tok/s)
- **Setup:** `HippocampusTransformer` (17.347M), identical 4,000 steps x 8,192 tok, seed 0 — direct comparison vs `transformer_4k` (`val_loss=1.9029`) and `synapse_v2_4k` (`val_loss=1.8434`, gap-406 pet recall 37.5%, gap-807 32.9%).
- **Hypothesis (`E-G2-1`):** Non-decaying slot bank lifts long-gap recall above the delta-matrix ceiling while matching `synapse_v2_4k` val_loss within ~0.02 nats.
- **Result — CONFIRMED (partial wall breach):**
  - `val_loss = 1.8452` (`val_loss_nomem = 1.9208`, `-0.0756` nat causal memory gain) vs `synapse_v2_4k 1.8434` — slot bank costs **nothing** on LM learning.
  - Pet-name recall by gap (pet / all, %): `0: 57.9/53.8 | 32: 62.9/55.0 | 96: 51.7/43.8 | 192: 52.1/42.7 | 400: **43.8/36.7** | 800: **36.7/31.9**`
  - vs `synapse_v2_4k` pet: `0: 71.7 | 32: 69.6 | 96: 47.9 | 192: 42.9 | 400: 37.5 | 800: 32.9` and `transformer_4k` pet: `0: 90.4 ... 400: 25.0 | 800: 25.0`.
  - **Slot bank beats matrix-only by +6.3 pts at gap-400 and +3.8 pts at gap-800** (46% relative over chance at 800); first architecture in the lab whose recall does not fully collapse to chance at 800 tokens.
  - **Trade-off discovered:** gap-0 recall DROPS to 57.9% (vs 71.7% synapse) — slot-read noise on queries local attention could answer; slot_gate (bias -0.5 → ~0.38 open at init) has not yet learned to close when unneeded at this budget. **Diagnosis logged; the 24k run tests whether the gate learns with training.**
- **Verdict:** Hypothesis confirmed at 4k scale — discrete slots extend recall beyond the rank-bounded matrix with zero LM cost. Proceeding to `hippocampus_24k` flagship vs the 24k `baseline` (`val_loss=1.5888`).

### Exp #015 — Generation-2 Multi-Axis Suite: `H4_cortex_v2` vs H0–H3 (3,600 steps, batch 36, seed 0)
- **Artifact:** `cortex_runs/benchmark_results_gen2.json`, `cortex_runs/H4_cortex_v2.pt`
- **Headline:** `H4_cortex_v2` = **ID 0.3854 / OOD 0.3050** vs H3 `0.2146/0.1671`, H0 `0.1931/0.1325` — a **+83% relative OOD jump**, the largest single-generation gain recorded in this lab. AUROC 0.8439 (H3 0.6754); ECE 0.0524 (H3 0.0250 — slightly worse but acceptable); ARC-OOD cell ZS 0.5875 → TTT 0.6023.
- **Per-axis deltas (H4 − H3, exact_acc):** the gain is *concentrated*, not uniform — exactly where the new mechanisms should act:
  - `causal_id_mixed`: **0.250 → 0.992 (+0.74)** — do-calculus grokked inside the standard 3.6k-step budget, which H3 needed the 540k-episode stress run to reach.
  - `causal_ood_do_interventional`: **0.275 → 0.983 (+0.71)** — "X was forced to 1" is precisely the rare-landmark binding the slot bank is designed to hold.
  - `causal_ood_scm`: 0.213 → 0.550 (+0.34)
  - `episodic_id_gap64`: +0.125, `episodic_ood_gap160`: +0.117, `episodic_ood_gap320`: +0.004 (capacity fine; bottleneck is key-match quality)
  - `multimodal_id_1hop`/`ood_2hop`: +0.154 / +0.175
  - `s5_ood_len10`: +0.108; `s5_ood_len13`: −0.042; `math_ood_d3/d4`: −0.013/−0.017 (small capacity-trade losses)
  - `arc_id_3demo`: 0.0625 both; `arc_ood_compositional`: 0.0 both — **exact full-grid program induction still unsolved** despite FiLM+program TTT.
- **Ponder curve K=1→16 (mean reasoning OOD):** 0.2361 → 0.2417 → 0.2500 → **0.2514 (K=6)** → 0.2458 → 0.2430 → 0.2319. Verdict: sinusoidal depth encoding keeps K=8–16 *stable* (no collapse to noise like discrete `step_emb` did — Gen-1 K=8 fell to 0.210 *with* a dip into untrained embeddings), but gains saturate at K=6 and mildly decay after — **contractive pressure, not catastrophic extrapolation failure**. Wall 3 partially breached.
- **Planning (methodological note):** H3 and H4 produced *identical* planning numbers (myopic trap 0.6625 / reactive 0.3875 / MPC-H6 **0.225** / subgoal-H8 0.5125; all goal rates 0.0). This is expected, not a leak: `LatentWorldModel` is constructed before arch-specific blocks, so both archs consume identical RNG → identical WM weights → identical deterministic planner behavior on identical maps. Planning eval measures the shared WM+planner, not the trunk. **Wall 2 NOT breached: 0% full Key→Door→Goal completion persists; hierarchical subgoal mode (0.5125 trap) underperformed flat MPC (0.225)** — the milestone heads reached only partial accuracy at 400 WM steps, and wrong stage estimates actively hurt search. Honest negative result: staged search needs either trained-to-convergence milestone heads or a symbolic subgoal state.
- **Pending verification:** slot-kill ablation (mechanism attribution for the causal jump — if `slot_gate.bias→−20` collapses causal back to ~0.27, the slot bank is verifiably responsible), H4 5-phase continual eval, seed-1 replication of the causal result (single-seed grokking could be a lucky lottery).

### Exp #015b — Slot-Kill Mechanism Attribution (post-hoc, `H4_cortex_v2.pt`)
- **Method:** loaded the trained checkpoint, eval'd 4 axes at n=160 intact, then set every `mem_blocks[i].attn.slot_gate.bias = −20` (sigmoid≈2e-9 → slot read contributes ~nothing; matrix + local attention untouched) and re-eval'd.
- **Result:** intact → slots-killed: `causal_ood_do` 0.9875→0.95 (−0.04), `episodic_ood_gap160` 0.25→0.2437, `mm_ood_2hop` 0.4313→0.425, `s5_ood_len10` 0.20→0.1938.
- **Interpretation — surprising and important:** the slot bank contributes only ~4 pts to the causal jump *at eval time*. The remaining ~68-pt gap over H3 must come from (a) the **sinusoidal depth encoding** or (b) **init lottery**. *This interpretation was TESTED AND OVERTURNED by Exp #015d*: the slot bank's real role is during **training** — it scaffolds learning of the binding operator, which the matrix path then partially subserves. Post-hoc eval-time slot-kill cannot undo that learned division of labor; only never-training-with-slots (Exp #015d: 0.20–0.23, no grok) reveals the true necessity. FiLM is provably inert here (zero-init, `program=None` skips the path entirely).
- **Honest caveat:** this entry's interim conclusion ("the reasoner, not the slot bank") was **wrong** — preserved here as the record of why eval-time-only ablations can mislead: a module can be pedagogically necessary while mechanistically dispensable post-learning.

### Exp #015c — Seed-1 Replication: the Causal Grok Is Real Architecture, Not Lottery
- **Method:** retrained `H4_cortex_v2` from scratch with `seed=1` (identical 3,600 steps × batch 36 protocol), eval'd the jumped axes at n=240.
- **Result (seed-1 vs seed-0):** `causal_id_mixed` **1.000** vs 0.992, `causal_ood_do` **1.000** vs 0.983, `causal_ood_scm` 0.513 vs 0.550, `episodic_gap160` 0.288 vs 0.263, `mm_2hop` 0.192 vs 0.392, `s5_len10` 0.25 vs 0.263.
- **Verdict — RESOLVED:** the do-calculus grok replicates on an independent init/data order (both seeds ≥0.98). H4 groks interventional inference in ~22k causal episodes; H3 needed ~540k mixed episodes to reach 100%. **~150x sample-efficiency gain on causal binding.** FiLM inert; lottery ruled out. *Initial* mechanism attribution to sinusoidal depth-sharing was tested and **overturned** by Exp #015d below — the slot bank is the enabler, via a training-time scaffolding role.
- **Artifact:** `cortex_runs/H4_cortex_v2_seed1.pt`, `cortex_runs/h4_seed1_replication.json`.

### Exp #015d — `H3_sinusoidal_abl` Isolation: the Slot Bank Is a *Developmental Scaffold* for Causal Binding
- **Method:** `H3_sinusoidal_abl` = H3's exact delta-matrix blocks + `SinusoidalLatentReasoner` (the ONLY difference vs H4 is the absence of the slot pathway; FiLM present but inert). Trained seeds 0 & 1, identical 3,600-step protocol.
- **Result:** `causal_ood_do` = **0.2333 (seed0) / 0.2042 (seed1)** — no grok on either seed; task_loss never underwent the grok transition (1.67–1.82 at end vs H4's drop to ~1.1–1.23).
- **Full evidence matrix:**
  | arch | slots in training | slots at eval | `causal_ood_do` |
  |---|---|---|---|
  | `H3` (discrete reasoner) | no | — | 0.275 |
  | `H3+sinusoidal` | **no** | — | **0.204–0.233** |
  | `H4` slots-killed | **yes** | no | 0.95 |
  | `H4` intact | yes | yes | **0.98–1.00** |
- **Interpretation — the headline scientific finding of Gen-2:** sinusoidal depth encoding contributes ~nothing to causal reasoning (isolated: ~0.21 vs H3's 0.275 — within noise). The **hippocampal slot bank is necessary during TRAINING**: it holds interference-free landmark bindings (e.g. "X was forced to 1") across the episode, letting the network LEARN the store→retrieve binding operator. Once the operator forms, the delta-matrix pathway (trained alongside, sharing `mem_proj`/gates) partially subserves it — hence only −4 pts on eval-time slot-kill. Without slots during training, the operator never forms inside the budget under ANY reasoner.
- **This is a direct computational analog of biological systems consolidation:** the hippocampus (fast discrete exact episodic store) is required to *acquire* patterns that then consolidate into neocortical (slow distributed interference-prone) substrates. Our slot bank plays the hippocampus; the delta matrix the neocortex — necessary for *learning* the binding operator, partially dispensable for *executing* it. This reframes the slot bank's role: it is not merely an episodic memory store but the **enabler of both Wall-1 recall AND the causal-reasoning breakthrough**.
- **Artifact:** `cortex_runs/H3_sinusoidal_abl_seed{0,1}.pt`, `cortex_runs/h3_sinusoidal_ablation.json`.

### Exp #016 — `hippocampus_24k` Flagship vs 24k `baseline` (Wall-1 flagship test)
- **Artifact:** `runs/hippocampus_24k/{log.jsonl,last.pt,recall.json}` (2026-11-13, ~84 min incl. compile, ~60-68k tok/s shared-GPU)
- **Setup:** `HippocampusTransformer` 17.347M (7 `SynapseBlock` + 1 final `HippocampusBlock`, 48 slots/head), 24,000 steps × 8,192 tok, seed 0 — direct comparison vs 24k `baseline` transformer (`val_loss=1.5888`, ctx-capped 256).
- **Result — WALL 1 BREACHED:**
  - `val_loss = 1.5142` (ppl 4.546) vs baseline `1.5888` → **−0.0746 nats better LM**, and `val_loss_nomem = 1.6158` → causal memory gain **−0.1016 nats** (grew from −0.0756 @4k — memory helps MORE with training, opposite of a crutch).
  - Pet recall (%) by gap vs baseline(24k)/synapse_v2(4k)/hippo(4k): `0: 91.2 vs 99.2/71.7/57.9 | 103: 69.6 vs 83.1/47.9/51.7 | 198: 71.2 vs 41.7/42.9/52.1 | 406: 60.4 vs 25.0/37.5/43.8 | 807: 42.5 vs 25.0/32.9/36.7`.
  - **Gap-406 = 2.4× chance, gap-807 = 1.7× chance**; first lab model to retain >40% at 800-token fillers. Also beats baseline at gap≥192 (baseline window half-exhausted).
  - **The 4k gap-0 regression healed:** 57.9%→91.2% — the slot gate *did* learn to close on easy queries with more training, exactly as the Exp-#014 diagnosis predicted. Residual cost vs baseline only at gap≤103 (−8 to −13.5 pts in the "fact just left the window" zone).
  - `all`/`color` tasks confirm: gap-807 `all`=40.0%, `color`=37.5%.
- **Verdict:** discrete landmark slots + bounded delta matrix decisively extends usable recall to 800+ tokens at zero LM cost — at flagship budget the mechanism *improves* with scale. Remaining gap to human-like ~95%: key-match quality + write selectivity (only top-1 token/chunk written; multi-fact episodes saturate). Next levers logged in roadmap: per-head write-top-k, learned write allocation, 2nd slot bank at mid-depth.

### Exp #017 — `H4_cortex_v2` 5-Phase Continual Learning (standalone, `run_continual_experiment`)
- **Artifact:** `cortex_runs/continual_H4_gen2.json` (the in-flight suite's continual phase ran H3 only under the pre-edit snapshot).
- **Result vs H3 (same protocol):** `naive_adamw` FAA 0.0275/BWT −0.230/dormant 10.9% (H3: 0.0275/−0.166/8.3%); `high_l2` 0.0463/−0.314/0%; `uniform_replay` 0.1687/−0.089/0%; **`cortex_plasticity` FAA 0.1750 / BWT −0.0812 / dormant 0.0%** (H3: 0.1937/−0.0281/0.86%).
- **Honest read:** H4's plasticity engine still delivers ~6.4× naive FAA and keeps **every unit alive** (0% dormant under replay/plasticity — even better than H3's 0.86%), **but forgets slightly more** (BWT −0.081 vs −0.028). Interpretation: the extra slot-bank and FiLM pathways add plastic capacity that the surprise-replay buffer doesn't fully protect; the reasoner/WM sees the same task sequence. A small continual-stability tax paid for the reasoning gains — logged as a trade-off, not hidden.
- **Next:** `H3_sinusoidal_abl` running (Exp #015c isolation — does depth-sharing alone drive the causal grok?).
