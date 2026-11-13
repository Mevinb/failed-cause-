# AGI Research Knowledge Base & Fundamental Bottleneck Analysis

**Status:** Living Research Artifact (Deliverable 1 — Phases 1 & 2)  
**Laboratory:** `agi-exi` Autonomous AGI Research Laboratory

---

## 1. First-Principles Decomposition of General Intelligence

An agent interacting with an open-ended, partially observable, non-stationary universe $\mathcal{E}$ receives observations $o_t \in \mathcal{O}$, emits actions or communications $a_t \in \mathcal{A}$, and seeks to maximize cumulative multi-task goal achievement under severe resource bounds:
- **Finite Working Bandwidth:** Exact all-to-all relational comparison scales as $O(W^2)$ for window size $W$.
- **Unbounded Interaction Horizon:** Total lifetime $T \gg W$, requiring $O(1)$-cost persistent episodic and semantic state.
- **Bounded Sample Complexity:** New tasks and rules arrive with $k \in \{1, \dots, 10\}$ examples, prohibiting million-step re-training per task.
- **Variable Computational Depth:** Some situations require $O(1)$ reflexive lookup; others require $O(K)$ sequential constraint propagation, search, or simulation steps.
- **Causal & Non-Stationary Environment:** Correlations shift when confounders or interventions occur; learning new regularities must neither erase prior skills nor degrade future learnability.

Any candidate AGI architecture must therefore provide **five irreducible computational primitives**:
1. **Structural-Sensory Factorization:** Separating invariant abstract operators from transient entity bindings.
2. **Multi-Timescale Memory:** Exact local relational working memory + interference-resistant fast-weight episodic memory + consolidated slow semantic weights.
3. **Adaptive Latent Computation:** Variable-depth recurrent state refinement anchored to perceptual/working context.
4. **Action/Operator-Conditioned World Modeling:** Predicting latent state transitions and causal interventions alongside epistemic surprise.
5. **Metacognitive Self-Monitoring & Plasticity Control:** Calibrated uncertainty/error estimation governing halting, self-correction, memory writes, and continual consolidation.

---

## 2. Why Current AI Systems Struggle: Mechanistic Root Causes

| Observed Failure Mode | Underlying Mathematical / Architectural Root Cause |
|---|---|
| **Lack of genuine abstraction & brittle OOD generalization** | Entangled token/feature representations conflate *structural role* with *surface identity*; without explicit iterative variable binding, models fit local interpolation manifolds that collapse outside the convex hull of training data. |
| **Catastrophic forgetting AND plasticity loss** | Single-timescale dense weights $W$: gradient steps $\Delta W = -\eta \nabla_W \mathcal{L}_{\text{new}}$ overwrite subspaces critical for prior tasks (forgetting), while prolonged Adam/SGD updates inflate weight norms and kill ReLU/SiLU units, collapsing Hessian rank (plasticity loss). |
| **Failure on long-horizon state tracking & planning** | Fixed-depth Transformers and diagonal SSMs lie in $\mathsf{TC}^0$; autoregressive greedy token emission commits irreversibly at each step without latent lookahead rollout or backtracking. |
| **Inability to maintain persistent world state** | Sliding/cropped KV caches discard everything beyond $W$ tokens (100% information loss, verified in `runs/baseline/recall.json`: 25.0% chance accuracy at gap $\ge 400$); naive recurrent states mix unrelated episodes across boundaries without surprise/episode gating. |
| **Confusing correlation with causality** | Maximum-likelihood next-token/observation training optimizes $P(Y \mid X)$ (Pearl Rung 1). Under unobserved or observed confounders $Z \to X, Z \to Y$, conditional conditioning fails to sever $Z \to X$ under $do(X=x)$. |
| **Sample inefficiency on novel rules** | Adaptation happens either purely in static activations (no weight update, bounded by fixed depth) or via slow outer-loop SGD requiring thousands of examples. |
| **Unreliable self-evaluation & hallucination** | Softmax token entropy measures *aleatoric next-word ambiguity*, NOT *epistemic validity* of a latent reasoning chain; models lack an explicitly trained verifier/error-prediction circuit decoupled from surface fluency. |

---

## 3. Phase 2: The Ten Most Important Unresolved Scientific and Engineering Problems

### Problem 1: Breaking the Fixed-Depth ($\mathsf{TC}^0$) Bottleneck Without Brittle Verbalized Chains
How can a neural architecture dynamically allocate $K(x) \in [1, K_{\max}]$ sequential, non-linear computational steps in continuous latent space—remaining stable when unrolled beyond training depth ($K_{\text{test}} > K_{\text{train}}$)—without collapsing to degenerate fixed points or requiring human-curated Chain-of-Thought supervision?

### Problem 2: Interference-Free, Episode-Aware $O(1)$ Associative Episodic Memory
Linear/matrix fast-weight memories ($M \in \mathbb{R}^{d_v \times d_k}$) have rank bound $\min(d_k, d_v)$. How should write gates, key orthogonalization (Delta rule), episode-boundary erasure, and surprise signals interact so rare salient facts persist across thousands of tokens without cross-episode bleed or subspace collision with RoPE local attention?

### Problem 3: Dual Resolution of the Stability-Plasticity Dilemma
How can an agent continually acquire new task distributions online while simultaneously achieving **non-negative Backward Transfer** (no catastrophic forgetting of earlier tasks) *and* **preserved Forward Plasticity** (no dormant-neuron/rank collapse over sequential tasks)?

### Problem 4: Variable Binding and Systematic Compositional Generalization
How can continuous vector architectures bind novel symbols/objects to abstract algebraic or spatial operators so rules learned on restricted arities/lengths/vocabularies transfer zero-shot to unseen symbol compositions and longer horizons?

### Problem 5: Interventional Causal World Modeling ($P(Y \mid do(X))$ vs. $P(Y \mid X)$)
How can a representation learner discover modular causal mechanisms from mixed observational and interventional experience, accurately distinguishing correlation from intervention on unseen causal graphs?

### Problem 6: Decision-Time Planning in Learned Latent Spaces Under Sparse Rewards & Traps
How should an architecture couple fast amortized policy priors with deliberate decision-time trajectory search (MPC / MCTS over a learned latent world model) so it avoids irreversible dead-ends without combinatorial explosion?

### Problem 7: Sample-Efficient Test-Time Rule Induction (High Chollet Fluid Intelligence)
Given only 2–4 demonstration input-output pairs of an unseen transformation (as in ARC), what inner-loop inference or test-time plasticity mechanism reliably induces the minimal underlying transformation program rather than surface pixel/token heuristics?

### Problem 8: Calibrated Epistemic Self-Evaluation and Autonomous Error Correction
How can a model predict the probability that its own candidate solution is wrong ($\hat{p}_{\text{correct}}$ with low Expected Calibration Error and high AUROC), and use that internal error signal to trigger targeted latent re-computation that measurably improves accuracy?

### Problem 9: Unified Multimodal Grounding Across Spatial Grids, Symbolic Graphs, and Language
How can 2D spatial/visual structures, discrete symbolic states, actions, and natural language share a common latent workspace without destroying geometric topology (e.g., 2D neighborhood invariance) during serialization?

### Problem 10: Safe, Empirical, Non-Collapsing Recursive Self-Improvement
How can an autonomous research system generate architectural/hyperparameter hypotheses, test them under strict compute budgets, detect Goodharting or overfitting, and commit only verified Pareto improvements with full reversibility?
