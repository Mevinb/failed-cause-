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

### Exp #018 — Wall-2 Decomposition & Breach: Grounded Symbolic Beam Search Solves 90% of Held-Out Maps
- **Artifact:** `cortex_runs/planning_gen2_eval.json`, `cortex/planning.py` (`_symbolic_step`, `select_action_symbolic`, `symbolic_decoded`/`symbolic_oracle` eval modes)
- **Failure chain traced first (the method mattered more than the fix):**
  1. `milestone_head` all-3-exact = **92.75%**, `obs_dec` cell-acc = **86.3%** — the heads were *not* the problem.
  2. Multi-step `step_latent` rollouts: after ~2 imagined steps, only ~23% of decoded boards contain *exactly one* agent; by step ~7 the decoded board has **no agent at all**, and when one reappears it is frozen at the start position. **Imagined latents leave the encoder manifold almost immediately** — re-grounding (decode→clean→re-encode every 2 steps) could not rescue boards that decode to garbage.
  3. First symbolic-planner draft scored beams by *raw return* and let trapped states keep expanding → **oracle only 3/10, 7 traps** — planner-logic bugs, not perception bugs. Three fixes: (a) trapped = terminal per env rule, (b) best-action tracked by *penalized* score, (c) door cell enterable in BFS once `has_key` (stepping on it opens it — otherwise distance-to-door saturates at the penalty floor).
- **Result (40 held-out maps, seed 60000+, H4 checkpoint + 1,200-step WM):**
  | mode | goal_rate | trap_rate | mean_return |
  |---|---|---|---|
  | myopic_H1 | 0.0 | 0.65 | −0.283 |
  | reactive_policy_H0 | 0.0 | 0.175 | −0.187 |
  | world_model_mpc_H6 | 0.0 | 0.275 | −0.204 |
  | hierarchical_subgoal_H8 | 0.0 | 0.375 | −0.236 |
  | reground_symbolic_H12 | 0.0 | 0.275 | −0.211 |
  | symbolic_decoded_H16 | 0.0 | 0.50 | −0.257 |
  | **symbolic_oracle_H16** | **0.90** | **0.0** | **+2.28** |
- **What "oracle" actually is (not privileged):** `cells0 = obs[3:39] − COLOR_BASE`, `flags = (obs[2] − COLOR_BASE) → (has_key, door_open)` — the observation tokens the model legitimately receives. `KeyDoorTrapEnv` is fully observable, so obs-parsing IS the honest perception pathway; positions (`key/door/goal`) are read from the parsed board, not `env.*` attributes.
- **Verdict — WALL 2 BREACHED at the planning layer, with a precise decomposition:**
  - The **search** was never the wall: exact deduplicated beam search in a verified symbolic state space (state = agent_pos × milestone_flags; dedup exact, not latent-quantized) solves 90% of held-out maps with near-optimal 7–14 step paths and **zero** trap deaths — vs 0% goal completion for every latent-space planner at any horizon.
  - The real wall was **state-abstraction fidelity**, and it has two named sub-walls now: (a) *perception gap* — learned `obs_dec` at 86.3% cell-acc is insufficient for planning (critical cells — agent/key positions — must be ~exact; decoded-mode gets 0%), (b) *imagination gap* — `step_latent` unrolls drift off-manifold within ~2 steps, so long-horizon latent MPC is fundamentally capped regardless of scoring.
  - **Architecture lesson (thinking outside the box):** don't make the learned latent transition more accurate at long horizons — *move the state* into a domain where correctness is verifiable. Decode/parse to a symbolic board, then run exact discrete search with milestone-consistent transition rules. This is the planning analog of the Wall-1 finding: discrete symbolic structure (slots there, symbolic states here) beats bounded continuous representations where precision matters.
- **Remaining 10%:** maps unsolved under H=16 + beam 64 — likely layouts where the optimal path exceeds episode cap or the beam prunes the detour; not investigated further this round.
- **Honest negatives preserved:** `symbolic_decoded` (learned perception) = 0% — decoder fidelity remains unsolved; `reground_symbolic` (decode→clean→re-encode) = 0% — off-manifold latents decode to unusable boards; latent `subgoal` mode still underperforms flat MPC.

### Exp #019 — `H4_cortex_v2` Autonomous Evolution Cycle (all mutants rejected — baseline stands)
- **Artifact:** `cortex_runs/evolution_archive_H4_cortex_v2.jsonl`, `evo_h4.out`
- **Setup:** `autonomous_loop.py --arch H4_cortex_v2` — Gen-0 champion (val_ood 0.3261) + 4 falsifiable candidates at identical 3,600-step budget, safety-gated (`max_mem_norm`), auto-rollback on negative delta.
- **Result:** **0/4 mutations promoted** — the Gen-2 baseline defended itself:
  | gen | mutation | val_ood | Δ | verdict |
  |---|---|---:|---:|---|
  | 0 | `gen0_H4_cortex_v2_base` | 0.3261 | — | champion (held-out OOD 0.3050) |
  | 1 | `ablation_no_surprise_gate` | 0.2989 | −0.027 | rejected (control — surprise gate still contributes, esp. mm_2hop 0.43→0.23) |
  | 2 | `mut1_finer_microchunk_mem48` | 0.3208 | −0.005 | rejected (finer writes + rank-48 matrix ≈ neutral, small negative) |
  | 3 | `mut2_deep_recursion_verifier` | 0.1813 | −0.145 | rejected (**ponder_steps=6 + micro_chunk=8 destroyed the causal grok**: causal_do 1.0→0.3 — over-deeper untrained depth + finer chunks broke the binding scaffold) |
  | 4 | `mut3_denser_slots_mc8` | 0.2812 | −0.045 | rejected (**n_slots=96 + micro_chunk=8 hurt** — write-capacity is NOT the OOD bottleneck) |
- **Interpretation:** two informative negatives. (a) `mut3` shows that scaling slot-bank capacity/granularity alone does NOT buy OOD — the ~4-pt write-selectivity residual is an LM-recall issue, not an OOD-reasoning issue, so `n_write_per_chunk` is tested in `lab/` (LM recall), not here. (b) `mut2` re-confirms that the causal grok is *fragile to the training curriculum*: deeper ponder + finer chunks in the same run killed the grok — the scaffold's benefit is path-dependent, not just architectural. Champion `gen0_H4_cortex_v2_base` stands with held-out OOD 0.3050.

### Exp #020 — Closing the Perception Gap: Weighted-CE Decoder Fidelity Lifts `symbolic_decoded` 0%→62.5%
- **Artifact:** `cortex_runs/H4_cortex_v2_decft{,W}.pt`, `cortex_runs/planning_gen2_decft_eval.json`
- **Setup:** after Exp #018 named the "perception gap" (learned `obs_dec` 86.3% cell-acc → decoded-perception symbolic planning = 0%), test whether it is a training-budget artifact or a capacity ceiling — dedicated fine-tune of `obs_enc`+`obs_dec` on real obs only, cross-entropy weighted ~12× on rare landmark cells (key/door/goal/trap ≈1/36 cells each) + 3× on the agent.
- **Result — budget artifact, not a wall:**
  - In-sample decode: 13.5% (untrained-mix checkpoint) → **100%** cell-acc, 100% exactly-one-agent, ~600 steps. Rare cells are the weak link under plain CE: held-out per-type key 66% / door 69% / goal 64% / trap 78% vs common cells ~99% — class imbalance, exactly the cells that define subgoals.
  - Weighted-CE held-out decode: **99.25% cell-acc**, landmark cells 93–96%, one-agent 96.8%.
  - `symbolic_decoded` planning (40 held-out maps, suite-checkpoint encoder + fine-tune): **goal 62.5% (25/40), trap 2.5% (1/40), return +1.557** — vs 0% for the un-sharpened decoder and 90% for obs-parsed oracle.
  - **Encoder quality matters as much as the decoder:** the same weighted-CE pass after a fresh 1,200-step JEPA `train_world_model` retrain gave only **5% (2/40)** in `evaluate_planning` — the suite's longer curriculum yields a better-structured `encode_obs` latent for decoding; decode fidelity cannot rescue a poorly-shaped encoder.
- **Full Wall-2 planning table now:**
  | perception / search | goal | trap |
  |---|---:|---:|
  | latent MPC any horizon (H6/H8/H12) | 0% | 10–38% |
  | symbolic search on un-sharpened `obs_dec` | 0% | 50% |
  | symbolic search on JEPA-retrained enc + weighted-CE decode | 5% | 10% |
  | **symbolic search on suite-encoder + weighted-CE decode** | **62.5%** | **2.5%** |
  | symbolic search on obs-parsed board | 90% | 0% |
- **Interpretation:** the neural end-to-end pipeline `obs → encode_obs → obs_dec → symbolic board → exact beam search` now demonstrably works (62.5% > 0%), closing the perception gap from "impossible" to "partially closed" — remaining ~27-pt gap to oracle is residual landmark-cell errors + encoder-representation quality. The imagination gap (latent rollout fidelity) remains fully open.

### Exp #021 — Imagination-Gap Distillation Fails: Continuous Latent Cannot Internalize Discrete Dynamics
- **Artifact:** `cortex_runs/step_latent_distilled.pt`
- **Setup:** the cleanest possible test of the "imagination gap" — distill the *exact* symbolic transition `_symbolic_step` into `step_latent` on 1,200 boards × actions with the sharpened encoder/decoder (Exp #020) frozen: `(z_t, a) → MSE(ẑ_{t+1}, encode(symbolic_next))`. If the continuous transition could represent discrete dynamics, distillation should drive faithful rollouts; if the gap is representational, it should fail regardless of supervision quality.
- **Result — REPRESENTATIONAL, not a data/budget issue:**
  - distill MSE converged 0.09→0.025 (the network fits the continuous regression), but rollout fidelity did NOT translate: strict whole-board match dies at ~1/12 steps; even the lenient **agent-position-only criterion = 72.6% at ONE step**, landmark+agent rollout 0.78/12, **0%** 12-step faithful trajectories.
  - The failure mode: ẑ lands "close" to the true z in MSE but decodes to a *different* board — the encoded z space has continuous slack that one-cell decode errors exploit; there is no pressure toward the discrete board manifold.
- **Interpretation:** the imagination gap is a **representation-geometry mismatch** — exact discrete transition dynamics want a discrete state, and a continuous `z + trans(z,a)` update cannot hold one. This *strengthens* the Exp-#018 conclusion rather than weakening it: grounded symbolic search is the right planner for verifiable discrete domains. For learned latent dynamics to imagine faithfully, the **state itself must be discrete** (a VQ/slot world-model state — the same discrete-structure-wins pattern as the Wall-1 slot bank and Wall-2 symbolic states). Future direction logged: a slot-latent world model whose state IS the symbolic board.

### Exp #022 — Top-k Slot Writes (n_write_per_chunk=3): FALSIFIED at 4k Gate
- **Artifact:** `runs/hippocampus_w3_4k/` (val_loss 1.8452, nomem 1.9207); `runs/hippocampus_w3_4k/recall.json`
- **Setup:** matched-budget A/B vs `hippocampus_4k` (identical config, seed 0, 4,000 steps; only `n_write_per_chunk` 1→3). Hypothesis: the gap≤103 residual exists because top-1-per-chunk writes drop co-occurring facts from multi-fact 16-token chunks — writing the top-3 surprising tokens should close it. Run **uncompiled** (~24k tok/s vs w1's ~63k compiled): the Python kw-loop triples the traced write graph and `torch.compile` never finished warmup.
- **Result — FALSIFIED:** w3 ≤ w1 at **every** gap — all-metric recall gap0/38/103/198/406/807: 54.2/53.5/40.8/41.0/33.5/31.5 vs w1's 53.8/55.0/43.8/42.7/36.7/31.9. val_loss bit-matched (1.8452 both), nomem gain identical (−0.0755 vs −0.0756 nat).
- **Interpretation:** write-*rate* is not the Wall-1 residual's bottleneck — consistent with the autonomous-loop `mut3` rejection (96 slots + micro_chunk 8 *hurt* OOD −0.045). More writes/chunk churns the 48-slot bank 3× faster → more evictions + merge collisions on landmarks → slightly worse recall uniformly. The residual is a write-*selection* problem (which tokens, learned allocation), not a write-quantity problem. Gate FAILED → no 24k w3 flagship; `hippocampus` stays `n_write=1`. Code kept (`n_write_per_chunk` config) for future learned/allocation variants.

### Exp #023 — Discrete-State World Model: the Imagination Wall IS Breachable
- **Artifact:** `cortex/discrete_wm.py`, `cortex_runs/discrete_wm_eval{,2,3}.json`, `cortex_runs/discrete_wm.pt`
- **Setup:** direct test of Exp #021's prescription — a *discrete* latent state (36 cells x 8 types + 4-value milestone flag, exactly the parsed obs board) with a learned transition `T(s,a)`, trained on parsed ground-truth state pairs (10,425 transitions, 600 envs). Rollouts hard-argmax each step so the state stays on the discrete manifold BY CONSTRUCTION. Held-out eval: 40 maps, imagine 14 steps along the expert plan, score exact-state / agent-pos / landmark-set fidelity vs true parsed states. Extra `exact_oracle` track: true parsed start state -> trans (encoder-free attribution).
- **Stage-1 finding (negative, then diagnosed):** flat-MLP encoder+trans hit 100%/99.6% in-sample but rolled out at **0% everywhere** — and the `exact_oracle` track was ALSO 0% at depth 1. Root cause was NOT exposure bias and NOT the discrete state: the flat MLP **memorized positional statistics of the training layouts** (erased all 4 landmarks on held-out maps). Two fixes: shared per-position encoder head (copy-task -> exact generalization) and a **conv-equivariant transition trunk** (grid dynamics are local rules), plus true scheduled sampling (trans fed its own 1-step predictions).
- **Result — WALL BREACHED (learned imagination now works):**

| depth | continuous z+trans (Exp#021) | MLP discrete | **conv-discrete** |
|---|---|---|---|
| 1 | 72.6% agent / 0% traj | 0% | **95% exact, 100% agent** |
| 3 | ~0% | 0% | **80%** |
| 6 | — | 0% | **51%** |
| 12 | **0%** | 0% | **31.5%** |

- `exact == exact_oracle` at every depth — the encoder now generalizes exactly, so the FULL neural pipeline (obs -> encode -> imagine) matches the oracle-discrete track.
- **Interpretation:** Exp #021's conclusion is refined — the imagination wall was never *continuous vs discrete* alone; it needed BOTH (a) a discrete state that can't drift off-manifold AND (b) an equivariant transition architecture that learns the local dynamics *rule* instead of positional statistics. ~4-6 steps of near-exact imagination covers the proven H=6 MPC horizon. Residual ~5%/step trans error is the next budget (more envs / longer training / deeper scheduled-sampling chains).
- **Follow-ups in-experiment:** (i) 8x oversampling of flag-change transitions FALSIFIED — landmark fidelity rose but movement broke (agent_d1 0.175): batch distribution distortion cost more than milestone coverage gained (`discrete_wm_eval4.json`); (ii) scaling data 10.4k -> 34.5k transitions (2000 envs) without reweighting lifted fidelity to **exact_d1=1.00, d3=0.90, d6=0.62, d12=0.53** with a ~55-60% plateau instead of collapse (`discrete_wm_eval5.json`); (iii) **first fully-LEARNED planner**: obs -> shared-head encode -> conv-T beam search (dedup + BFS shaping + trap head) solves **35% (14/40) held-out maps, 0% traps, +1.01 mean return** (`cortex_runs/discrete_wm_planning.json`) — vs symbolic_oracle 90%, symbolic_decoded 62.5%, all latent-MPC modes 0%. The imagination breach translates into real planning; remaining gap to oracle is residual `trans` error at milestone transitions and depth >8.
- **Next:** wire learned `T` into the main `planning.py` eval as a first-class mode; deeper scheduled-sampling chains; test whether the discrete-state trick lifts `step_latent` for partially-observable domains.

### Exp #024 — Learned Write Allocation (ST-estimator pick): Improves LM, Regresses Recall
- **Artifact:** `runs/hippocampus_la_4k/` (val_loss **1.8398**, nomem 1.917, −0.0772 nat mem gain); `runs/hippocampus_la_4k/recall.json`
- **Setup:** the write-*selection* lever the Exp-#022 falsification pointed to. Pick becomes `argmax(surprise + alloc_head(x))` with the discrete choice trained via a straight-through estimator (hard pick fwd, softmax-grad back). Zero-init head re-zeroed after `_init_weights` (it re-randomizes all Linears). Same budget/seed as `hippocampus_4k`/`w3_4k`. Run uncompiled — ST ops bloat the traced graph like w3's did.
- **Result — MIXED, honest negative for the residual:** all-metric recall **lower than w1 at every gap** (45.8/50.2/38.3/35.4/32.7/28.3 vs 53.8/55.0/43.8/42.7/36.7/31.9); val_loss slightly **better** (1.8398 vs 1.8452, −0.0054 nat).
- **Interpretation:** the alloc head learns the criterion gradient *can* teach — writes that help immediate next-token prediction — which is NOT the rare-associative-fact retention the recall test measures. "Will this token be queried 400 steps later" is not inferable from `x` at write time, so end-to-end learned selection defaults to LM-relevant patterns. Combined with Exp #022 (rate falsified), the write-policy design space is now narrowed to: (a) retrieval-aligned auxiliary objectives (write for *retrievability*, not surprise/LM), (b) mid-depth second bank, (c) better merge/eviction policy. Bonus finding from debugging: `surp` saturates ≈1.0 at init — the argmax-surprise policy starts near-arbitrary and differentiates with training (median 0.94, max 1.16 in the trained model).

### Exp #025 — Discrete-WM v6 Scaling + Manifold-Projection Battery: Only Transition Quality Moves Planning
- **Artifact:** `cortex_runs/discrete_wm_eval6.json` (v6 fidelity), `discrete_wm_eval7_factored.json` (projection A/B), `discrete_wm_eval8_surg.json`, `cortex_runs/discrete_wm.pt` (v6 weights)
- **Setup:** scaled the Exp-#023 recipe — 6,000 envs / 103,417 transitions (3x), 8,000 steps (2x), scheduled-sampling chains to depth 3. Then, on the frozen v6 weights, ran a battery of *inference-time* interventions aimed at the residual error, each isolating one manifold constraint: `pin` (landmarks position-pinned + learned flag), `fact` (pinned + milestone-derived flag), `surg` (per-cell argmax + exactly-one-agent enforcement only), `staged` (raw + 3xpopcount(flag) milestone-progress bonus in beam scoring).
- **v6 fidelity (40 held-out maps):** exact_d1=1.00, d3=0.90, d6=**0.72** (v5: 0.62), d12=0.37; teacher-forced trans_cell 0.998 / flag 0.996.
- **v6 learned planning — the headline: 62.5% (25/40), 0% traps, +1.62 mean return, 14.2 steps.** The earlier 35% figure was the v5 checkpoint; data+SS scaling nearly doubled learned-dynamics planning (0% → 35% → 62.5%).
- **Failure localization (the corrected residual):** agent-position mismatches only **2/431** steps — but **110/431 (~26%) imagined boards contain NO agent at all** (evaporation, not drift). Milestone transitions no longer disproportionate (33.8% fail vs 30.5% moves) — data scaling fixed the v5 milestone problem. First-divergences land on ordinary moves 19:6.
- **All four eval-time interventions FALSIFIED (≤ raw):** plan goal rates — raw **62.5%**, staged 62.5%, surg 60%, pin 42.5%, fact 42.5%. Fidelity strictly worse under full projection (agent_d6: 0.82 raw vs 0.62 factored).
- **Interpretation:** (i) *exposure-bias in reverse* — scheduled sampling robustified the transition to its own argmax-error distribution; projecting onto the legal manifold creates an input distribution the model never trained on, so every projection variant degrades it. (ii) The beam is robust to agentless nodes (dedup collapses them but siblings carry the plan) — agent revival alone adds nothing. (iii) Scoring-side shaping (staged milestones) adds nothing either — the residual is purely transition quality. **Only the model matters now.**
- **Planning failure anatomy (15 maps):** 7 stall pre-key (all optimal-length ≥12 — long-horizon dilution), 7 open the door then miss the goal (rarest flag=3 regime), 1 stalls at door. Zero traps.
- **Next (Exp #026, launched):** agent-existence auxiliary loss (soft-count → 1) under scheduled sampling — teaches "exactly one agent" as a *learned* self-correcting invariant rather than a projected one — plus 12k envs / 206k transitions / SS-4 / 10k steps.

### Exp #026 — Agent-Existence Aux Loss + Full-Board Dedup: the bug fix beats the loss
- **Artifact:** `cortex_runs/discrete_wm_eval7.json` (v7), `discrete_wm.pt` (v7 weights), `discrete_wm_eval9_v8.json` (v8 = v6-recipe replication)
- **Setup:** two independent changes conflated in v7 (206k trans, SS-4, agent-existence aux loss `MSE(Σsoftmax(agent), 1)`), followed by a pure planner fix — **full-board dedup** replacing the `(agent_pos, flag)` key inherited from the symbolic planner, where it was correct because `(ag,flag)` IS the complete state. On noisy imagined boards it collapses distinct states — beam diversity destroyed.
- **Aux loss result — partial mechanism validation, no planning gain:** agent evaporation 25.6% → 20.6%, but position-mismatches 2 → 33 (the count penalty keeps *an* agent alive, less constrained *where*). Fidelity crossover: worse early (exact_d6 0.51 vs v6 0.72), better deep (exact_d12 0.47 vs 0.37). Planning ~flat.
- **Dedup fix result — largest single-change gain since conv-equivariance:** same v7 weights, same process, replicated twice: `(ag,flag)` dedup **35%** → full-board dedup **60%** (+25 pts, 0% traps). The symbolic planner's dedup key silently discarded distinct imagined states.
- **Measurement caveat (logged for protocol):** rollout-eval results are *process-sensitive* — cudnn autotune picks different conv algos across processes, borderline argmax ties flip, chaotic beam rollouts diverge (eval7's in-training-process 57.5% did not replicate; within-process results are deterministic — agflag 35% and raw 60% each reproduced exactly in two processes). Treat within-process A/Bs as reliable; single cross-process numbers ±5pts at least.
- **Status:** v8 (v6 recipe: 103k trans / SS-3 / no aux) running to attribute aux-loss vs SS-depth under the fixed dedup. Checkpoints now versioned (`--ckpt-out`) — v6 weights were overwritten by v7 before the dedup fix was discovered.

#### Exp #026 addendum — v8 replication failure & the rollout-robustness lottery
- **Artifact:** `cortex_runs/discrete_wm_eval9_v8.json`, `discrete_wm_v8.pt`
- **Finding:** v8 = near-exact v6 replication (same 103,417-transition dataset — deterministic collection seed, same training seed/recipe: 8k steps, SS-3, no aux; only difference = GPU nondeterminism in training) produced a model with **identical teacher-forced metrics (trans_cell 0.9858, flag 0.9531) but catastrophically broken rollouts**: exact_d1 0.75 → 0 by d4, `exact_oracle` 0 by d4, planning 0%. Eval code exonerated by re-running on v7 weights (reproduced its recorded fidelity exactly).
- **Interpretation:** the v6 62.5% planning result is ONE DRAW from a high-variance training distribution — teacher-forced accuracy cannot distinguish a rollout-robust basin from a memorization basin. The v8 run's final logged batch spiked (loss 0.421 at step 8000 vs 0.011 at 7500) — late-training instability may be the proximal cause, but the deeper lesson is that **rollout-fidelity model selection is mandatory**: claims must be evaluated at the checkpoint selected for the property being claimed.
- **Fix implemented:** `train()` now supports `--ckpt-every N` periodic checkpoints + `select_checkpoint()` picks by held-out rollout fidelity (16 maps × depth 8, mean exact+agent) — converts the lottery into selection. `--ckpt-out` gives versioned checkpoints.
- **Honest state of the art for the discrete-WM planner:** oracle-symbolic 90%; learned-dynamics best-confirmed = **60%** (v7 weights + full-board dedup, replicated 2×); v6's 62.5% stands but rests on an unverifiable pre-fix measurement. Variance itself is now a named open problem (SS robustness basin vs memorization basin).
- **Running:** v9 = v6 recipe + checkpoint selection (8 candidates) — tests whether selection alone rescues the brittle draws.

#### Exp #026 addendum-2 — fidelity≠planning: checkpoint selection needs the claimed metric
- **Artifact:** `cortex_runs/discrete_wm_eval10_v9.json`, `discrete_wm_v9{,_step*}.pt`
- **The surprise:** v9's fidelity-selected checkpoint (step5000, probe-score 0.839 — the best rollout-fidelity model ever measured: d6 0.72/d9 0.66/d12 0.58) plans at only 25-35%. The run's FINAL checkpoint (step8000, fidelity-rejected at 0.791) plans at **57.5% raw / 45% agflag** — fidelity and planning-capability *anti-correlate* across checkpoints in this regime. Branch-wise: reward heads identical (|err| ~0.016, goal-step ~1.97), random-action fidelity equal-or-better — the decoupling is real, not a head artifact.
- **Mechanistic read:** the beam needs *branch-ranking* accuracy (which action's imagined outcome is best), which keeps improving with total training steps (sel16 planning on disjoint seeds: 0%@3k → 37.5%@8k — monotone), while exact-board fidelity on optimal paths saturates ~5k then oscillates. Selection must therefore use the *claimed* metric (a planning proxy on a disjoint seed set — used 70000+ vs eval 50000+) — or simply train longer.
- **Dedup-key verdict refined:** full-board dedup is consistently right ON MATURE models (+25pts on v7, +12.5 on v9-8k); it hurt only the under-trained step5000 model. `raw` (full-board) is now the default comparison.
- **Current confirmed best:** v7 weights raw **60%** (replicated 2x), v9-step8000 raw **57.5%** — the ~60% plateau is real; the recipe→model path remains high-variance (v8 = 0%) so checkpoint sets + planning-proxy selection are the protocol going forward.
- **Open:** why planning competence keeps growing after exact-fidelity saturates — likely branch-calibration/refinement of rare transition regions; measured next via per-checkpoint beam-quality probes.

### Exp #027 — v10: plan-metric checkpoint selection at 16k steps; within-run basin lottery exposed
- **Artifact:** `cortex_runs/discrete_wm_eval11_v10.json`, `discrete_wm_v10{,_step2000..16000}.pt`, `discrete_wm_eval11b_v10_compare.json`, `discrete_wm_eval11c_branchrank.json`
- **Setup:** v7 recipe scaled — 12k envs / 206,710 transitions / 16,000 steps / SS-4 / agent-existence aux loss; checkpoints every 2,000 steps; `select_checkpoint(metric="plan")` on 16 disjoint selection maps (seed0=70000).
- **Within-run instability observed live:** loss spike 3.673 at step 10,500 (trans_cell crashed 0.998→0.754); full recovery by step 14,000 (0.997, loss 0.021). The v8-style catastrophic draw is not only across runs — it happens *inside* a run.
- **Selection scores (sel16):** step2000 = 0.000 | **step4000 = 0.438** | step6000–12000 = **0.000 across the board** | step14000 = step16000 = 0.312. → step4000 selected, copied to canonical `discrete_wm_v10.pt`.
- **"Planning grows monotone in steps" (the v9 inference) FALSIFIED:** competence appears by 4k, is *destroyed* in the 5k–12k instability window, and only partially re-emerges after recovery (0.312 < 0.438). Within-run basin-hopping, not growth.
- **Selected checkpoint (step4000) on 40 held-out maps — best-ever fidelity AND solid planning:** exact_d1 1.00 / d3 0.925 / d6 0.744 / d9 0.625 / d12 0.526 / d14 0.75 (d14 re-inflation = absorbing goal state); `exact == exact_oracle` everywhere (encoder remains perfectly OOD). **plan_raw 55% / agflag 47.5%, 0% traps, +1.44 mean return.** step16000 for reference: raw 55% / agflag 25%.
- **Verdict on plan-metric selection:** works as designed — sel-maps correctly ranked live checkpoints above dead ones (43.8% → 55% eval, no overfit), and step4000 dominates step16000 on agflag (+22.5). On raw, eval ties 55/55 despite sel-maps ranking them apart — 16-map selection has ±2-map noise. Protocol stands; consider larger sel set.
- **Plateau intact:** confirmed learned-dynamics planning remains ~55–62.5% across every lever tried (data scale, SS depth, aux loss, steps, projections, staged scoring, dedup key, checkpoint selection). Neither selection nor 2× training breaks it.

#### Exp #027 addendum — branch-ranking probe: scoring is NOT the residual (Simpson's correction)
- **Probe:** 1,200 random action prefixes (len 2–8) × 10 maps; imagined discounted return (reward head − 4·p_trap, beam's own formula) vs true env return.
- **Pooled Spearman = −0.51** — apparent catastrophic anti-ranking. **But it's a Simpson's-paradox artifact:** prefixes that grab the key THEN trap score +0.475 true (milestone reward banked before death) yet −2.88 imagined (trap detected → −4/step). ~10% crossover pairs flip the pooled sign. Within-class the ranking is **healthy: +0.855 non-trapping (n=649), +0.888 trapping (n=551)**; zero spurious trap detections on safe prefixes; p_trap = 1.000/0.000 separation at every depth; reward head tracks true within ~0.01.
- **Consequence:** reward/trap heads and branch scoring are *calibrated and correct* — they are not the residual. (The beam's −4·p_trap compounding IS conservative vs env's banked-milestone reward structure, but directionally right for avoiding real traps: 0% trap rate everywhere.)
- **Per-step oracle-agreement probe:** the learned planner's first action matches symbolic-oracle on **35/40 (87.5%)** initial states; failures span 18 maps. Compounding arithmetic: ~0.87 per-step agreement over ~12-step plans ⇒ ~20–40% end-to-end success — the observed 55% sits inside the predicted band. **The plateau is a compounding-agreement phenomenon, not a scoring, dedup, or selection problem.**
- **Remaining levers, ranked by mechanism:** (a) amortized value/policy head trained on oracle-BFS labels (bypasses rollout scoring — but shades toward distilling the oracle rather than testing imagination); (b) beam width/horizon scaling (tests search-budget); (c) anything that pushes per-step agreement 87%→95%+.

### Exp #028 — hippocampus_nov at the 4k gate: the first non-falsified write lever (small, directional)
- **Artifact:** `runs/hippocampus_nov_4k/` (val_loss 1.8453, nomem 1.9206), `runs/hippocampus_nov_4k/recall.json`
- **Setup:** `novelty_gate=True` — write pick scores `surprise + ‖v_cand − bank_read(k_cand)‖` (reconstruction novelty vs the current slot bank; detached, zero added params). Matched 4k budget, seed 0, vs w1 (`hippocampus_4k`), w3, LA.
- **Result — directional win at retrieval gaps, first variant to not regress:** recall gaps 32/96/192/400/800 = **56.0/44.6/42.9/38.5/32.1** vs w1 55.0/43.8/42.7/36.7/31.9 → **+1.0/+0.8/+0.2/+1.8/+0.2 (positive at all 5)**; gap-0 = 51.2 (−2.6, the only regression — novelty defers writes the bank can already partially reconstruct, and near-window facts read fine anyway). LM-neutral (1.8453 vs 1.8452). Contrast: w3 negative at all 5 gaps, LA negative at all 5 gaps and worse (−5 to −4).
- **Interpretation:** consistent with the mechanism — surprising tokens the bank *cannot already reconstruct* are the ones worth storing, raising slot utilization for distinct facts. Effect size ~+1pt (inside the ±4.4 rotation CI), so per protocol this is a *hypothesis confirmed directionally, not yet a win*: seed-1 replication launched (`hippocampus_nov_s1`); if direction holds, a 24k matched run decides whether the small edge compounds like w1's did (gap-406: 36.7%@4k → 60.4%@24k).
- **Note on gap-0:** −2.6 at gap-0 echoes w1's own early-training gap-0 dip (57.9%→91.2% healed by 24k as the slot gate learns to close on easy queries) — worth watching whether it heals identically.

#### Exp #028 addendum — seed-1 does NOT replicate; the 4k recall metric is noise-limited
- **Artifact:** `runs/hippocampus_nov_s1/` (val_loss 1.839, nomem 1.9147), `recall.json`
- **Result:** nov seed-1 recall = 53.3/51.0/37.1/32.7/30.6/28.7 (gaps 0/32/96/192/400/800) — at or below w1-seed0 at every retrieval gap, and far below nov-seed0 (e.g. gap-400: 30.6 vs 38.5). The seed-0 "+~1pt at all retrieval gaps" direction **did not replicate**.
- **Verdict revised:** novelty selection is *not falsified* but *not confirmed* — the effect (if any) is smaller than the 4k recall metric's seed-to-seed variance (±5–8pts at some gaps; nov_s1 pet@0 68.8% vs s0 57.1% illustrates the scale). Two-seed protocol is now mandatory for small-effect claims at the 4k gate. `hippocampus_s1` (w1 on seed 1) running to bound w1's own seed variance — converts "nov_s1 is a bad draw" vs "nov is harmful" into a measurable comparison.
- **Meta-lesson (second occurrence):** a small consistent-direction effect at 4k (first w3's near-tie, now nov's +1pt) is within run variance; the recall eval resolves ~±5pts across seeds, so gate claims need either multi-seed or the 24k budget where effects compound (w1: gap-406 36.7→60.4).

#### Exp #028 addendum-2 — w1 seed-variance bounds the whole question: nov is NEUTRAL, gate is noise-limited
- **Artifact:** `runs/hippocampus_s1/` (val_loss 1.8392, nomem 1.915), `recall.json`
- **w1 across seeds:** gaps 96/192/400/800 = s0 43.8/42.7/36.7/31.9 → s1 35.6/36.9/29.4/27.5 — swings of **−8.2/−5.8/−7.3/−4.4 pts on the SAME architecture** between seeds.
- **Final verdict:** nov vs w1 = +1.8 (s0) / +1.2 (s1) at gap-400 — *inside* w1's own ±5–8pt seed noise. Novelty selection is a **neutral variant** at 4k resolution: not falsified (unlike LA's consistent ~−5) but unproven. Write-selection remains open; the 4k recall metric cannot resolve <±5pt effects — decisive tests need the 24k budget (where w1's gap-406 compounded 36.7→60.4) or a higher-resolution probe.
- **Standing protocol correction:** any future write-policy variant claiming <5pt effect must launch ≥2 seeds at the gate.

### Exp #029 — progressed-regime data augmentation BREAKS the planning plateau: fully-learned = oracle (90%)
- **Artifact:** `cortex_runs/discrete_wm_eval12_v11.json`, `discrete_wm_v11.pt` (= step8000, canonical), `discrete_wm_v11_step*.pt`, `logs/dwm11.log`
- **Setup:** Exp #027's named residual — mid-episode action-ranking noise in progressed states (flag=3 agreement 41.9% vs flag=0 50.7% at t0) — pointed at *data scarcity*: every training episode starts at flag=0, so flag≥1 regimes are the rarest transitions. Fix: `collect_progressed_transitions` — same map layouts, episodes START from `has_key`/`door_open` with the agent repositioned; +46,306 transitions (18% of 253,016 total). Otherwise identical to v10 (16k steps, SS-4, aux-agent, ckpt-every-2000, plan-metric selection).
- **Training signature:** noisy early phase (trans_cell 0.91–0.98 through ~6.5k — the distribution-shifted mix is harder to fit), sharp basin convergence at ~7000 (0.936→0.994), late-phase instability at 11k/12.5k/14k/15.5k with recoveries — same within-run lottery dynamics as v10.
- **Selection curve (sel16):** 0.000×3 (steps 2–6k) → **0.812 (8k)**, 0.688 (10k), 0.562 (12k), 0.312 (14k), **0.812 (16k)** — the converged phase plans at 2–4× any previous checkpoint. step8000 selected.
- **Headline — 40 held-out maps:** `plan_raw` **goal_rate 0.90 (36/40), trap_rate 0.0, mean_return +2.29, mean_steps 12.05** — **matches the symbolic oracle exactly** (90%, 0% traps, +2.28). `plan_val` 85% (the learned ValueNet now slightly underperforms the model's own dynamics — the auxiliary potential is no longer needed). Fresh-process replication on the checkpoint file: **0.90 / 0.0 / +2.293 — confirmed**.
- **Fidelity (same ckpt):** exact d1 1.00 / d3 0.925 / d6 0.718 / d9 0.688 / d12 0.579 / d14 0.75-absorbing — comparable to v10's best curve, so the gain is not from fidelity but from *where* the model is accurate (progressed regimes).
- **Interpretation:** the 55–62.5% plateau was a **coverage gap**, not a capacity, scoring, search, or selection problem. Beam lookahead over ~16 steps spends most of its probability mass in flag≥1 states that were ~5–10% of natural trajectories; giving those regimes ~18% of transitions fixed the ranking. Every falsified lever stays falsified — they were all correct diagnoses of non-causal suspects; the causal variable was training-distribution coverage of the states the planner actually visits.
- **Provenance caveat logged:** a duplicate v11 process (orphaned when only the bash wrapper was killed) raced on the same checkpoint filenames; the surviving step8000 artifact was verified byte-stable and reproduces 0.90 in an independent process. Kill semantics for future launches: kill the python child, not the `bash -c` wrapper.
- **Status:** Wall-2's learned-dynamics component is now at the oracle ceiling on this task family (90%). Remaining Wall-2 scope: scaling to richer dynamics, partial observability, and the latent-imagination variant (where the state isn't a directly-parsed board).

### Exp #030 — Adaptive halting probe: convergence ≠ correctness; dz-halting saves 2.7x compute AND beats full depth
- **Artifact:** `cortex/probe_halting.py`, `cortex_runs/halting_probe.json`; `H4_cortex_v2.pt` (eval-only, trained model), `arch.py` (+`dz` in H4 forward return)
- **Setup:** H4 trained with randomized deep supervision (k_train in {2..6}); ran reasoner to K=16 on math_depth3 / s5_len10 / mm_2hop (n=120 each). Per-episode per-step: exact-acc, masked-mean dz, sequence conf. Halting rules: fixed-K, oracle-K (upper bound), dz<tau, conf>c, verifier-select (post-hoc argmax over all K).
- **Finding 1 — over-pondering DEGRADES OOD accuracy:** per-step acc peaks early then declines: s5 0.283@K4 -> 0.183@K16; multimodal 0.392@K6 -> 0.308@K12; math flat ~0.11. The contractive iteration converges — to the wrong fixed point. Wall-3 reframed: the problem is not stability at depth (sinusoidal solved that) but that **optimal answer depth is episode-specific and usually shallow**.
- **Finding 2 — dz is a convergence signal, not a correctness signal:** dz falls monotonically (3.50 -> 0.39 over K=16 on s5) while accuracy falls — the state keeps contracting after the answer has been reached and corrupted.
- **Finding 3 — dz-threshold halting is a legitimate adaptive halter:** s5 tau=0.64 -> halts mean-K 5.8 at acc 0.275 vs fixed-K16 0.183 (+50% acc, 2.7x less compute); multimodal tau=0.66 -> mean-K 3.0 at 0.383 vs 0.325. Halting at convergence strictly dominates running the full depth.
- **Finding 4 — verifier_select ≈ fixed-K16** (0.183 on s5): the learned confidence head does not rank steps correctly OOD — it inherits the training distribution's step preferences.
- **Finding 5 — oracle-K gap is the real remaining wall:** s5 oracle 0.442 vs best rule 0.283; mean oracle depth ~1.9 steps. Episodes differ widely in optimal depth, but NO eval-time signal recovers it — dz says "converged" not "correct". The named next lever: a **halting head** trained to predict per-step answer correctness (the per-step supervision already exists in training) — i.e., learn the correctness signal dz can't provide.
- **Status:** Wall-3 adaptive halting: dz-rule CONFIRMED as compute-saver/accuracy-preserver; oracle-gap analysis motivates trained halting-head (Exp #031).

### Exp #031 — Trained halting head: partial oracle-gap recovery (multimodal only)
- **Artifact:** `cortex/halting_head.py`, `cortex_runs/halting_head.json`
- **Setup:** small MLP `HaltHead` on 6 scalar per-step features (dz, conf, k/K, entropy, dz-slope, conf-slope) trained 600 iters on ID splits (pos-weighted BCE, ~29% positive rate), frozen H4. OOD eval (n=160/task): halt rules vs oracle/dz/conf.
- **Result:** multimodal: `halt_thr0.7` 0.481 vs oracle 0.550, best fixed rule 0.431-0.456 — recovers ~50% of the gap; s5: all rules 0.20-0.22 vs oracle 0.375 — no recovery; math: no headroom (oracle 0.12). The head is the single best non-oracle rule only where ID features transfer (multimodal).
- **Interpretation:** scalar features carry usable correctness signal on some tasks but not s5 — where the optimal-depth decision needs state-level information (what the verifier tried and failed to exploit OOD). Verdict: PARTIAL — the oracle-K gap is real and partly recoverable by a learned signal, but scalar features are insufficient for the hardest case. Options: richer features (per-token conf trajectory, z-norm), or train the halt head jointly inside the model (a training change — deferred; GPU budget better spent on v12 replication + Wall-1 resolution).
- **Wall-3 ledger:** dz-halting CONFIRMED (compute-saver that beats full-depth OOD); trained-head halting PARTIAL (works where features transfer); full oracle-K recovery UNRESOLVED — needs state-aware halting, not scalar features.

#### Exp #029 addendum — structural OOD: learned planner holds up under 2x wall density
- `eval_learned_planning` gained `n_walls` (env parametrized 5 -> N walls; eval maps only emit solvable BFS-checked layouts). v11 step8000, fresh eval seeds (60000+):
  - 8 walls (1.6x training density): **80% goal, 0% traps, +1.98**
  - 10 walls (2x training density): **90% goal, 0% traps, +2.30**
- Interpretation: the conv-equivariant transition learned *local* grid rules (walls block; key->flag; door+key->open; goal->reward) that transfer to maze topologies outside the training distribution — the 90% is not a same-distribution artifact. 0% traps maintained under both densities.
- Caveat: n=40 per condition (within-binomial noise ~10-15%); the two conditions are consistent with a flat ~80-90% OOD band rather than a monotone trend.
