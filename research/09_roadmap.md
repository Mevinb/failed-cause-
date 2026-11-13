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
  - [ ] Per-head **top-k writes** (currently top-1/16-token chunk misses multi-fact episodes) — `n_write_per_chunk` config flag
  - [ ] **Learned write allocation** (replace argmax-surprise with a learned scorer) + mid-depth second slot bank
  - [ ] Multi-fact saturation & interference diagnostic (how many concurrent bindings before eviction errors dominate)
- [ ] **Offline Sleep-Phase Consolidation:** Distill high-surprise episodic buffer trajectories into slow structural weights during periodic offline consolidation cycles while resetting fast episodic matrices.

## Stage 1.5 — Gen-2 Verified Mechanisms & Open Attribution Work *(Inserted Post-Hoc, 2026-11-13)*
- [x] **Sinusoidal depth encoding** — stable ponder to K=16 (0.232 vs K6 0.251) and, per slot-kill + seed-1 replication, the **driver of the ~150× causal-grok sample-efficiency gain**. *Mechanistic follow-up:* `H3_sinusoidal_abl` (running) isolates depth-sharing vs slots definitively.
- [x] **FiLM program conditioning** — pathway verified functional; but exact ARC solve **0%** (falsified for exact program induction — needs discrete search, not gradient refinement).
- [ ] **Adaptive halting:** verifier/`dz`-norm-based per-episode stopping (ponder curve saturates at K=6 — spend compute only where it helps).
- [ ] **Subgoal world-model repair:** milestone heads need convergence or replacement with a symbolic subgoal state (has_key/door_open/goal decoded from z); current staged search underperforms flat MPC.

## Stage 2 — Deep Recursive Program Synthesis & Grounded World Simulation *(Medium-Term)*
- [ ] **Discrete-Continuous Hybrid Workspace:** Augment `IterativeLatentReasoner` with differentiablediscrete slot attention and soft execution traces so induced rules on 2D grids (ARC-AGI-2) can compose arbitrary $k$-step programs with test-time verification search over `ttt_adapter` hypotheses. *(Gen-2 evidence: gradient-refined FiLM programs insufficient — add a library-of-primitives enumeration layer guided by the reasoner prior.)*
- [ ] **Hierarchical Multi-Step JEPA World Models:** Extend `LatentWorldModel` from single-step transitions $\hat{z}_{t+1} = f(z_t, a_t)$ to temporal-abstraction options / subgoal predictors $\hat{z}_{t+\Delta} = F(z_t, g)$, enabling planning over $10^2$–$10^3$ action steps without exponential beam growth.
- [ ] **Interventional Causal Discovery Agent:** Move from passive observational/interventional episodes to *active* experimental design—selecting interventions $do(X_i = v)$ that maximize expected information gain (epistemic reduction in structural graph entropy). *Leverage:* the now-verified fast causal grok makes this the most promising frontier — an agent that already does do-calculus can be asked to *choose* interventions.

## Stage 3 — Open-Ended Curriculum & Verified Self-Improvement *(Long-Term)*
- [ ] **Co-Evolution of Tasks and Architectures (POET / DGM Integration):** Couple `cortex/autonomous_loop.py` with an automated adversarial problem generator that synthesizes novel solvable compositional tasks right at the frontier of current verifier uncertainty ($\hat{c} \approx 0.5$).
- [ ] **Multi-Seed Statistical Gatekeeping & Formal Sandbox Guarantees:** Require $N \ge 5$ independent seeds, held-out adversarial OOD benchmark families, and automated mechanistic interpretability probes before promoting architectural mutations across generations.
- [ ] **Scaling Validation (50M $\to$ 350M $\to$ 1B+):** Test whether the architectural gains of `Cortex-AGI` (surprise-gated decoupled fast weights + input-anchored recursive workspace + calibrated epistemic verifier) compound or saturate with compute scale across code, formal math, and embodied multimodal environments.
