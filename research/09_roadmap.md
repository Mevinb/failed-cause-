# AGI Research & Engineering Development Roadmap

**Status:** Living Roadmap (Deliverable 9)  
**Laboratory:** `agi-exi` Autonomous AGI Research Laboratory

---

## Stage 0 — Mechanistic Dissection & Controlled Baselines *(Completed in Current Cycle)*
- [x] Patch `llama.cpp` (`Qwen3.5-9B`) with zero-copy structural, decay, residual-scaling, feedback, and recurrent state-surgery hooks.
- [x] Empirically establish the **Attention-Shortcut Starvation Theorem**: demonstrate double dissociation between sparse full-attention KV caches (100% of exact passkey/fact storage) and Gated-DeltaNet recurrent state (syntactic motor progression + language/affective style).
- [x] Establish iso-parameter (`17.3M`) from-scratch TinyStories language modeling & counterbalanced 0–800 token recall baselines (`runs/baseline`, `runs/*_4k`).
- [x] Build modular 11-subsystem `Cortex-AGI` (`SYNAPSE-1`) architecture and 8-domain ID/OOD cognitive evaluation suite with automated mathematical invariant tests (`cortex/`).

## Stage 1 — Multi-Timescale Episodic & Semantic Consolidation *(Active / Short-Term)*
- [x] Resolve micro-chunk decay scaling ($\alpha_{\text{chunk}} = \alpha_{\text{win}}^{1/8}$) and surprise-modulated sparse writes to prevent filler overwrite at 400–800+ token horizons.
- [x] **Hippocampal Sparse Landmark Slot Bank (Gen-2, WALL 1 BREACHED):** 48-slot/head content-addressable landmark memory alongside the delta matrix — `hippocampus_24k` `val_loss 1.5142 < baseline 1.5888`, gap-406 recall **60.4%**, gap-807 **42.5%** (vs 25% chance). *Remaining:* gap≤103 residual cost vs full-context baseline.
  - [x] Per-head **top-k writes** — `n_write_per_chunk` implemented, **FALSIFIED at 4k gate** (Exp #022): w3 ≤ w1 at every gap; write-rate isn't the bottleneck → next lever is *learned write allocation* (which tokens), not more writes
  - [ ] **Learned write allocation** (replace argmax-surprise with a learned scorer) + mid-depth second slot bank
  - [ ] Multi-fact saturation & interference diagnostic (how many concurrent bindings before eviction errors dominate)
- [ ] **Offline Sleep-Phase Consolidation:** Distill high-surprise episodic buffer trajectories into slow structural weights during periodic offline consolidation cycles while resetting fast episodic matrices.

## Stage 1.5 — Gen-2 Verified Mechanisms & Open Attribution Work *(Inserted Post-Hoc, 2026-11-13)*
- [x] **Sinusoidal depth encoding** — stable ponder to K=16 (0.232 vs K6 0.251). Its causal-attribution was tested and overturned (Exp #015d): `H3+sinusoidal` (no slots) fails to grok do() on 2 seeds (0.20–0.25) → the **slot bank is the training-time scaffold** that enables the causal breakthrough. Verified role of sinusoidal: K-extrapolation stability only.
- [x] **FiLM program conditioning** — pathway verified functional; but exact ARC solve **0%** (falsified for exact program induction — needs discrete search, not gradient refinement).
- [ ] **Adaptive halting:** verifier/`dz`-norm-based per-episode stopping (ponder curve saturates at K=6 — spend compute only where it helps).
- [x] **Subgoal world-model repair → RESOLVED by reframing (Exp #018, WALL 2 BREACHED at planning layer):** exact deduplicated beam search over *grounded symbolic* states (`agent_pos × milestone_flags`, terminal traps, BFS-to-subgoal shaping) solves **90% of held-out maps, 0% traps** — vs 0% for all latent planners. The wall was state-abstraction fidelity, not search. Two named sub-walls opened: (a) **perception gap** — largely closed (Exp #020: weighted-CE → 62.5% decoded planning); (b) **imagination gap** — BREACHED (Exp #023: discrete-state + conv-equivariant `T` → 35% fully-learned planning, faithful rollouts to depth ~6). *Remaining:* residual `trans` error at milestone transitions; partially-observable domains where obs-parsing alone is insufficient.

## Stage 2 — Deep Recursive Program Synthesis & Grounded World Simulation *(Medium-Term)*
- [ ] **Discrete-Continuous Hybrid Workspace:** Augment `IterativeLatentReasoner` with differentiablediscrete slot attention and soft execution traces so induced rules on 2D grids (ARC-AGI-2) can compose arbitrary $k$-step programs with test-time verification search over `ttt_adapter` hypotheses. *(Gen-2 evidence: gradient-refined FiLM programs insufficient — add a library-of-primitives enumeration layer guided by the reasoner prior.)*
- [x] **Discrete-state world model (Exp #023, imagination gap BREACHED):** `cortex/discrete_wm.py` — state = the parsed board (36 cells × 8 + flag), conv-equivariant transition, shared-head encoder, scheduled sampling (chain depth ≤3). Faithful rollouts `exact_d1=1.00/d3=0.90/d6=0.62/d12=0.53` held-out; first fully-learned planner (encode→conv-T beam search) **35% solve, 0% traps, +1.01** vs oracle 90%/decoded 62.5%/latent 0%. Falsifications en route: flat-MLP enc/trans memorize layout stats (0% OOD); 8× milestone oversampling breaks movement. *Open:* push `trans` fidelity (more envs, deeper SS chains), integrate as a `planning.py` eval mode, partially-observable domains where parsing fails.
- [ ] **Latent-dynamics distillation — SUPERSEDED by Exp #023:** continuous `z + trans(z,a)` distillation failed representational (72.6% @1 step, Exp #021); the discrete-state model is the working replacement. Longer-term alternative remains: options/subgoal predictors $\hat{z}_{t+\Delta} = F(z_t, g)$ skipping over steps entirely.
- [ ] **Interventional Causal Discovery Agent:** Move from passive observational/interventional episodes to *active* experimental design—selecting interventions $do(X_i = v)$ that maximize expected information gain (epistemic reduction in structural graph entropy). *Leverage:* the now-verified fast causal grok makes this the most promising frontier — an agent that already does do-calculus can be asked to *choose* interventions.

## Stage 3 — Open-Ended Curriculum & Verified Self-Improvement *(Long-Term)*
- [ ] **Co-Evolution of Tasks and Architectures (POET / DGM Integration):** Couple `cortex/autonomous_loop.py` with an automated adversarial problem generator that synthesizes novel solvable compositional tasks right at the frontier of current verifier uncertainty ($\hat{c} \approx 0.5$).
- [ ] **Multi-Seed Statistical Gatekeeping & Formal Sandbox Guarantees:** Require $N \ge 5$ independent seeds, held-out adversarial OOD benchmark families, and automated mechanistic interpretability probes before promoting architectural mutations across generations.
- [ ] **Scaling Validation (50M $\to$ 350M $\to$ 1B+):** Test whether the architectural gains of `Cortex-AGI` (surprise-gated decoupled fast weights + input-anchored recursive workspace + calibrated epistemic verifier) compound or saturate with compute scale across code, formal math, and embodied multimodal environments.
