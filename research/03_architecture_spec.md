# AGI Architecture Specification: Three Distinguishable Hypotheses & The `Cortex-AGI` (`SYNAPSE-1`) System

**Status:** Living Architectural Specification (Deliverable 3 — Phases 3, 4, 5)  
**Laboratory:** `agi-exi` Autonomous AGI Research Laboratory

---

## 1. Phase 3 Synthesis: Empirical & Theoretical Failure Analysis of Existing Architectures

Combining complexity theory (Merrill et al., 2024; Schölkopf et al., 2021; Dohare et al., 2024) with our direct empirical probes in `agi-exi`:

1. **Standard Dense Transformer (`H0`, e.g., `lab/model.py:Transformer`):**
   - *Hard horizon wall:* Exact 0% beyond-window information retention (`runs/baseline/recall.json`: 97.7% at gap 0 $\to$ **25.0% chance** at gaps 406 and 807).
   - *Bounded circuit depth ($\mathsf{TC}^0$):* Fixed $L$-layer pass allocates identical compute to trivial tokens and hard compositional deductions.
   - *Shortcut vulnerability & causal blindness:* Learns observational $P(Y \mid X)$ correlations; overwrites weights catastrophically under sequential task shifts.
2. **Pretrained Hybrid Gated-DeltaNet + Attention (`Qwen3.5-9B`, 24 GDN + 8 Full-Attention layers):**
   - *Empirical Double Dissociation (`results/state.jsonl`, `results/rewire.jsonl`):*
     - Wiping all 24 Gated-DeltaNet recurrent states to zero (`zero_all`) after encoding a passage **preserves 100% of exact fact recall** (`7391`, `Zorbak`, `Teal`), whereas dropping the final attention layer (`drop_31`) or middle attention layers (`no_attn_mid`) reduces passkey retrieval from **100% to 0.0%** while leaving parametric knowledge (`fact: 1.0`) intact.
     - Conversely, transplanting all 24 GDN states (`transplant_all`) transfers **zero** exact passkeys (`"1234"` hallucination), yet blending GDN states (`alpha=2.0`) potently transfers **language (`French`) and affective tone (`Angry`)**, while freezing early/mid GDN states causes immediate syntactic stutter collapse.
   - *Root Cause — Attention Shortcut Starvation:* When full attention spans the entire training sequence, softmax attention provides a zero-error shortcut for exact key-value retrieval, starving recurrent fast-weight layers of gradient pressure to learn high-precision episodic addressing.
3. **Naive Chunk-Level Associative Memory (`lab/model.py:MemoryTransformer`):**
   - Updates memory only once every 256 tokens while 84.7% of TinyStories episodes are $<256$ tokens (`val.bin` median = 183 tokens), never resets across `<|endoftext|>` (`id=0`) boundaries (injecting cross-story interference), and shares RoPE-twisted Q/K projections with position-invariant fast-weight memory.

---

## 2. Phase 4: Three Experimentally Distinguishable AGI Architecture Hypotheses

To avoid monolithic unfalsifiable designs, we formulate three distinct architectural hypotheses that make **divergent, experimentally testable predictions** under identical parameter and training budgets:

### Hypothesis H1 — `SynapseMem`: Multi-Timescale Surprise-Gated Fast-Weight Memory Architecture
- **Core Thesis:** The primary bottleneck to general intelligence is selective, interference-free, unbounded episodic binding. Pairing restricted local working attention with a **decoupled, episode-aware, surprise-gated Gated-Delta fast-weight matrix memory** updated at fine granularity ($C \ll T_{\text{episode}}$):
  $$e_t = v^{\text{mem}}_t - \alpha_t M_{t-1} k^{\text{mem}}_t, \qquad \beta_t = \sigma\!\left(w_\beta^\top x_t + \gamma_s \frac{\|e_t\|_2}{\sqrt{d_h}} + b_\beta\right), \qquad M_t = \alpha_t M_{t-1}(I - \beta_t k^{\text{mem}}_t (k^{\text{mem}}_t)^\top) + \beta_t v^{\text{mem}}_t (k^{\text{mem}}_t)^\top$$
  with exact episode reset $\alpha_t \leftarrow 0$ on boundary tokens will enable unbounded OOD context recall and rapid one-shot key-value binding, *without* variable-depth recurrence.
- **Predicted Signature:** Excels at long-horizon episodic recall (gaps 400–800+) and one-shot associative binding; degrades on multi-step compositional reasoning, $S_5$ state composition at high depth, and trap-laden planning.

### Hypothesis H2 — `LatentPonderWM`: Input-Anchored Iterative Latent Reasoning & JEPA World-Model Architecture
- **Core Thesis:** The primary bottleneck to general intelligence is fixed per-token circuit depth ($\mathsf{TC}^0$) and reactive decision-making without internal simulation. Replacing untied deep stacks with a **weight-tied, input-anchored recursive latent workspace** ($z^{(k)} = f_\theta(x_{\text{enc}}, y^{(k-1)}, z^{(k-1)}, k)$ for $k=1\dots K$) coupled with an **action/operator-conditioned latent world model** ($\hat{z}_{t+1} = g_\theta(z_t, a_t)$ trained in normalized representation space) enables variable test-time reasoning depth ($K_{\text{test}} > K_{\text{train}}$) and lookahead planning, *without* persistent fast-weight matrix memory.
- **Predicted Signature:** Excels at OOD multi-step compositional reasoning, abstract 2D rule refinement, causal intervention propagation, and lookahead planning; fails beyond-window episodic recall when working context is cropped.

### Hypothesis H3 — `Cortex-AGI` (`SYNAPSE-1`): Unified Neuro-Cognitive Workspace with Closed-Loop Surprise, Iterative World-Model, Epistemic Self-Evaluation, and Dual Continual Plasticity
- **Core Thesis:** General intelligence is neither memory alone (H1) nor recurrent depth alone (H2), but emerges from **closed-loop coupling** around a shared currency—**epistemic prediction error (surprise)**:
  1. Prediction error in associative/latent space drives **selective fast-weight episodic encoding** (solving the stability-plasticity and attention-starvation bottlenecks).
  2. Input-anchored **iterative latent refinement** updates both abstract workspace state $z^{(k)}$ and candidate solution $y^{(k)}$, escaping $\mathsf{TC}^0$ while preserving parameter efficiency.
  3. An explicit **Epistemic Self-Evaluator** $\hat{c}^{(k)} \in (0,1)$ trained to predict solution validity governs **adaptive compute allocation** (halting early on easy inputs, iterating + revising on hard/uncertain inputs) and scores **latent world-model rollouts** during planning.
  4. **Continual Plasticity Control** (fast episodic weights + surprise-prioritized replay + slow-weight shrink-and-perturb regularization) maintains both backward retention and forward learnability across sequential task shifts.
- **Predicted Signature:** Achieves Pareto dominance across *all* evaluation axes under matched parameter count, exhibits positive accuracy scaling with test-time iteration $K$, maintains calibrated uncertainty (low ECE, high error-detection AUROC), and prevents both catastrophic forgetting and plasticity loss.

---

## 3. Phase 5: Rigorous Comparative Evaluation of the Three Hypotheses

| Dimension | H0: Dense Transformer | H1: `SynapseMem` (Fast-Weight Delta) | H2: `LatentPonderWM` (Iterative + WM) | H3: `Cortex-AGI` (`SYNAPSE-1` Unified) |
|---|---|---|---|---|
| **Theoretical Complexity Class** | $\mathsf{TC}^0$ (fixed depth, window $W$) | Non-commutative state transitions; fixed depth per step | Arbitrary depth $O(K)$ in latent space; window $W$ | Non-commutative $O(1)$ episodic state **+** adaptive depth $O(K)$ |
| **Memory Complexity (Inference)** | $O(W \cdot L \cdot d)$ | $O(W_{\text{loc}} d + L H d_h^2) = O(1)$ in $T$ | $O(W \cdot L_{\text{core}} \cdot d)$ (weight-tied!) | $O(W_{\text{loc}} d + L_{\text{mem}} H d_h^2) = O(1)$ in $T$ |
| **FLOPs per Token / Decision** | Fixed $O(L d^2)$ | Fixed $O(L d^2 + L H d_h^2)$ (+2% FLOPs) | Adaptive $O((L_{\text{pre}} + K L_{\text{core}}) d^2)$ | Adaptive $O((L_{\text{mem}} + \bar{K} L_{\text{core}}) d^2)$, $\bar{K}$ gated by verifier |
| **VRAM Feasibility (6 GB RTX 4050)** | High (1.7 GB @ 17.3M) | High (2.0 GB @ 17.4M; fp32 $64\times 64$ state is tiny) | High (weight sharing *reduces* parameter memory; BPTT over $K \le 6$ fits easily) | High (1.9–2.4 GB at 17M LM scale; <0.6 GB at 4M multi-task scale) |
| **Scientific Foundations** | Vaswani et al. (2017) | Schlag et al. (2021), Yang et al. (2025), Behrouz et al. (2025), Whittington et al. (2020) | Geiping et al. (2025), Jolicoeur-Martineau (2025), Hafner et al. (2025), Assran et al. (2025) | Synthesis of CLS, Predictive Coding (Rao & Ballard 1999), GWT, Titans, TRM, DreamerV3, Dohare et al. (2024) |
| **Primary Failure Risks** | Zero beyond-window memory; no OOD depth; catastrophic forgetting | Attention shortcut starvation if $W_{\text{loc}}$ is too wide; fixed depth fails multi-hop composition | Representation drift / fixed-point collapse at high $K$ if un-anchored; forgets past episodes | Multi-objective loss balancing ($\mathcal{L}_{\text{task}} + \lambda_{\text{wm}} \mathcal{L}_{\text{wm}} + \lambda_{\text{ver}} \mathcal{L}_{\text{ver}}$); mitigated via unit-free normalized losses |

**Selection Decision (Phase 6):** We select **H3 (`Cortex-AGI` / `SYNAPSE-1`)** as our primary experimental architecture, while implementing **H0, H1, and H2 inside the exact same modular codebase** as controlled architectural ablations so every subsystem's causal contribution is measured rather than assumed.

---

## 4. Mathematical & Engineering Specification of `Cortex-AGI` (`SYNAPSE-1`)

```
                       +---------------------------------------------------+
  2D Grids / Vision -->| 1. MultimodalPerceptionEncoder (2D axial + patch)  |
  Text / Symbols    -->| 2. LanguageInterface (token + role/op embeddings)  |
                       +-------------------------+-------------------------+
                                                 | h_perc
                                                 v
                       +---------------------------------------------------+
                       | 3. DualMemorySystem + 9. RelationalSlotWorkspace  |
                       |  - Local RoPE Causal Working Attention (window W) |
                       |  - Decoupled Surprise-Gated Delta Episodic Memory |
                       |    M_t = a_t M_{t-1}(I - b_t k k^T) + b_t v k^T   |
                       |    (episode reset on EOT; surprise ||v - M k||_2) |
                       +-------------------------+-------------------------+
                                                 | h_0 (anchored context)
                                                 v
        +------------->+---------------------------------------------------+<------------+
        |              | 5. IterativeLatentReasoner (Weight-Tied Core)     |             |
        |   Feedback   |  u^(k) = Norm(z^(k-1) + W_inj[h_0; y^(k-1)] + e_k)|    Causal    |
        |  (k -> k+1   |  z^(k) = z^(k-1) + CoreBlock(u^(k), do_mask)      |  do(X=x)    |
        |   if c < tau)|  y^(k) = y^(k-1) + W_y z^(k)                      |  Graph Mask |
        |              +------------+--------------------------+-----------+             |
        |                           |                          |                         |
        |                           v                          v                         |
  +-----+---------------------------+---+          +-----------+-------------------------+----+
  | 8. EpistemicVerifier & 11. ErrorCorr|          | 4. LatentWorldModel & 6. MPCPlanner      |
  |  c^(k) = sigma(MLP([h_0,z^(k),y^(k),|          |  z_hat_{t+1}, r_hat_t = WM(z_t, a_t)     |
  |          ||z^(k) - z^(k-1)||_2]))   |          |  Beam/Tree lookahead scored by r + V(z)  |
  +-------------------------------------+          +------------------------------------------+
                                                 |
                                                 v
                       +---------------------------------------------------+
                       | 7. ContinualPlasticityEngine                      |
                       |  - Test-time fast-weight inner adaptation (TTT)   |
                       |  - Surprise-prioritized episodic replay buffer    |
                       |  - Shrink-and-Perturb / dormant-unit regeneration |
                       +---------------------------------------------------+
```

### Subsystem-by-Subsystem Equations & Invariants

1. **Perception & Multimodal Encoding (Subsystems 1, 2, 10):**
   - Discrete sequence tokens $x_{1:T}$ map to $E_{\text{tok}}(x_t) \in \mathbb{R}^d$.
   - 2D spatial grids $G \in \{0,\dots,C_g-1\}^{H \times W}$ map to cell/patch tokens $E_{\text{color}}(G_{r,c}) + E_{\text{row}}(r) + E_{\text{col}}(c) + E_{\text{grid\_role}}$, preserving exact 2D row/column alignment across input-output demonstration pairs and test grids.
2. **Dual Working & Surprise-Gated Episodic Memory (Subsystems 3, 9):**
   - Local Working Attention restricts causal attention to window $W_{\text{loc}}$ (preventing attention shortcut starvation when $T > W_{\text{loc}}$!).
   - Decoupled fast-weight projections $q^{\text{mem}}, k^{\text{mem}} = \text{L2Norm}(W_{qm} x), \text{L2Norm}(W_{km} x)$, $v^{\text{mem}} = W_{vm} x$ (computed in `fp32`).
   - Within micro-chunks of length $C$: read $r_t = g_{\text{read}} \odot (M_{c-1} q^{\text{mem}}_t)$; surprise $s_t = \|v^{\text{mem}}_t - M_{c-1} k^{\text{mem}}_t\|_2 / \sqrt{d_h}$; write gate $\beta_t = \sigma(w_\beta^\top x_t + \gamma_{\text{surp}} s_t + b_\beta)$; exact episode reset on `<|endoftext|>` (`idx == 0`) clears $M$ before cross-episode tokens are processed.
3. **Input-Anchored Iterative Latent Reasoning & Causal Surgery (Subsystems 5, 9, 11):**
   - Core transformer/mixer block with shared parameters $\theta_{\text{core}}$ applied $K$ times.
   - Input injection $W_{\text{inj}}[h_0 \,;\, y^{(k-1)}]$ guarantees gradient flow to $h_0$ at every depth $k$ and prevents fixed-point drift at $K_{\text{test}} > K_{\text{train}}$.
   - Under causal intervention $do(V_j = v)$, node $V_j$'s latent state is clamped to $E_{\text{val}}(v)$ at every recurrent step $k$ and incoming attention edges $\to V_j$ are masked, implementing Pearl's graph surgery $G_{\overline{V_j}}$ directly inside latent message passing.
4. **Latent World Model & Decision-Time Lookahead Planner (Subsystems 4, 6):**
   - Transition predictor $\hat{z}_{t+1} = \text{RMSNorm}(z_t + \text{MLP}_{\text{WM}}([z_t, E_{\text{act}}(a_t)]))$, reward predictor $\hat{r}_t$, terminal/trap predictor $\hat{d}_t$, and state value $V(z_t)$.
   - Trained with stop-gradient representation alignment $\mathcal{L}_{\text{WM}} = \|\hat{z}_{t+1} - \text{sg}(\text{RMSNorm}(z_{t+1}))\|_2^2 + \text{MSE}(\hat{r}_t, r_t) + \text{BCE}(\hat{d}_t, d_t)$.
   - At decision time, `LatentMPCPlanner` expands action sequences of horizon $H$ in parallel on GPU, pruning predicted trap states ($\hat{d} > 0.5$) and selecting the highest-return trajectory.
5. **Epistemic Self-Evaluation & Error Correction (Subsystems 8, 11):**
   - Verifier head outputs calibrated correctness probability $\hat{c}^{(k)} = \sigma(\text{MLP}_{\text{ver}}([h_0, z^{(k)}, y^{(k)}, \Delta z^{(k)}]))$, supervised against ground-truth exact match $\mathbb{I}[\arg\max \text{logits}^{(k)} == y^*]$ across intermediate steps $k=1\dots K$.
   - Adaptive halting stops at the earliest $k$ where $\hat{c}^{(k)} \ge \tau_{\text{conf}}$, or continues refining up to $K_{\max}$ and selects $\arg\max_k \hat{c}^{(k)}$ when confidence dips.
6. **Continual Plasticity & Test-Time Adaptation (Subsystem 7):**
   - **Test-Time Plasticity:** Given few-shot support pairs $(x_i^{\text{sup}}, y_i^{\text{sup}})$ of an unseen task/distribution, updates only lightweight fast-adapter/memory parameters via $N_{\text{TTT}}$ inner steps while keeping slow weights frozen—zero catastrophic forgetting by construction.
   - **Continual Consolidation:** Across sequential task phases $T_1 \to T_2 \to \dots \to T_M$, combines a compact **Surprise-Prioritized Episodic Buffer** (storing highest-loss exemplars per phase) with **Shrink-and-Perturb / Dormant-Neuron Reinitialization** (Dohare et al., 2024) to preserve both backward retention and forward plasticity.

---

## 5. Known Architectural Limitations & Honesty Statement

In strict adherence to scientific integrity:
1. `Cortex-AGI` (`SYNAPSE-1`) is a **research prototype**, NOT achieved human-level AGI.
2. Its fast-weight matrix memory per head has capacity $O(d_h^2)$ (bounded linear algebraic rank $d_h$), so storing millions of verbatim encyclopedic facts still requires an external non-parametric index or hierarchical memory bank.
3. Its causal intervention operator requires either explicit intervention markers (`do`) or interventional trajectories during training; inferring unobserved latent confounders from purely i.i.d. static data remains fundamentally non-identifiable without structural priors.
4. All capability claims are bounded to what is measurably demonstrated on held-out in-distribution and OOD benchmarks in Phases 6 and 7.
