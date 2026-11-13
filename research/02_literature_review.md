# Literature Review and Scientific Synthesis on Artificial General Intelligence

**Status:** Living Research Artifact (Deliverable 2)  
**Laboratory:** `agi-exi` Autonomous AGI Research Laboratory  
**Machine-Readable Companion:** `research/02_literature_database.json`

---

## 1. Defining and Measuring Progress Toward AGI

A rigorous research program requires a falsifiable target metric. Chollet (2019, `arXiv:1911.01547`) demonstrated via Algorithmic Information Theory that measuring static task *skill* confounds genuine intelligence with prior knowledge and training-set coverage: an experimenter can "buy" arbitrary benchmark accuracy by expanding data or hard-coding priors while generalization efficiency remains zero. Formally, general intelligence is **skill-acquisition efficiency** across a broad task scope under bounded priors and limited experience:

$$I(\text{system}) \propto \mathbb{E}_{T \sim \mathcal{T}}\left[\frac{\text{Generalization}(T) - \text{Chance}(T)}{\text{PriorInfo} + \text{Experience}(T) + \text{Compute}(T)}\right]$$

Consequently, throughout this laboratory's experiments:
1. Architectural candidates are evaluated under **strictly matched parameter, token, and step budgets**.
2. Every benchmark pairs in-distribution validation with **structural out-of-distribution (OOD)** tests (longer horizons, unseen entity bindings, confounded interventions, novel compositional rules, sequential task shifts).

---

## 2. Computational Expressivity Limits of Current Foundation Architectures

### 2.1 Fixed Circuit Depth and the $\mathsf{TC}^0$ Barrier
Standard decoder-only Transformers execute a fixed number $L$ of sequential layers per emitted token. Merrill & Sabharwal (2023) and Merrill, Petty, & Sabharwal (2024, ICML, `arXiv:2404.08819`) proved that both log-precision fixed-depth Transformers *and* diagonal/scalar-gated State-Space Models (S4, Mamba, Mamba-2) are bounded within uniform $\mathsf{TC}^0$ (constant-depth threshold circuits). As a mathematical consequence:
- Neither architecture can represent $\mathsf{NC}^1$-complete state-tracking problems—such as composing elements of the non-solvable symmetric group $S_5$, tracking general world states under sequential actions, or evaluating nested boolean/program expressions—beyond a length determined by fixed depth $L$, *unless* intermediate states are externalized token-by-token via Chain-of-Thought (CoT) or architectural transitions become non-commutative/depth-adaptive.
- Verbalized CoT is an expensive, brittle workaround: it consumes $O(T_{\text{CoT}}^2)$ KV-cache compute, forces continuous geometric or constraint-satisfaction states through discrete vocabulary bottlenecks, and fails without supervised/RL reasoning traces.

### 2.2 Escaping $\mathsf{TC}^0$: Non-Commutative Delta Transitions and Latent Recurrent Depth
Two complementary mechanisms provably expand computational expressivity:
1. **Input-Dependent Non-Commutative Fast-Weight Transitions (Delta Rule):** Updating a matrix state $M_t \in \mathbb{R}^{d_v \times d_k}$ via the gated Widrow-Hoff delta rule (Schlag et al., 2021; Yang, Kautz, & Hatamizadeh, ICLR 2025, `arXiv:2412.06464`):
   $$M_t = \alpha_t M_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$
   multiplies the state by input-dependent generalized Householder matrices $(I - \beta_t k_t k_t^\top)$, which do *not* commute across steps and can represent arbitrary permutation/finite-state transitions up to head dimension rank, while enabling targeted key-value overwriting ($M_t k_t \to v_t$ as $\beta_t \to 1$, $\|k_t\|_2 = 1$) and rapid episode erasure ($\alpha_t \to 0$).
2. **Adaptive Latent Recurrent Depth (Iterative Refinement):** Geiping et al. (NeurIPS 2025, `arXiv:2502.05171`, Huginn) and Jolicoeur-Martineau (2025, `arXiv:2510.04871`, Tiny Recursive Model / TRM) demonstrated that iterating a weight-shared recurrent block $K$ times in continuous latent space—critically **anchored** at every step to the encoded input $x$ (and current candidate hypothesis $y$)—allows a 7M-parameter model to outperform 100B+ parameter LLMs on constraint-satisfaction and ARC-AGI tasks. Notably, independent ablations by ARC Prize (2025) on Wang et al.'s Hierarchical Reasoning Model (`arXiv:2506.21734`) revealed that *iterative outer-loop refinement and deep supervision*, rather than multi-module hierarchy, accounted for virtually all reasoning gains—underscoring the necessity of ruthless ablation.

---

## 3. Neuroscience, Cognitive Architectures, and Multi-Timescale Memory

Human cognition and mammalian neurobiology converge on four organizational principles absent from monolithic Transformers:

1. **Complementary Learning Systems (CLS) & Relational Factorization:** McClelland et al. (1995) and Whittington et al. (2020, *Cell*, Tolman-Eichenbaum Machine) established that the brain separates **slow neocortical/entorhinal structural weights** (learning abstract transition algebras, syntax, and causal schemas) from **fast hippocampal associative bindings** (one-shot binding of arbitrary novel entities to structural roles). This factorization explains zero-shot relational transfer: an abstract rule learned on $\{A, B, C\}$ transfers immediately to $\{X, Y, Z\}$ because structural operators act orthogonally to fast entity bindings.
2. **Surprise-Modulated Episodic Encoding & Test-Time Memorization:** Predictive coding in cortex (Rao & Ballard, 1999; Friston, 2010) and neuromodulatory gating (noradrenaline/acetylcholine) allocate plasticity in proportion to **prediction error (surprise)**. Behrouz, Zhong, & Mirrokni (NeurIPS 2025, *Titans*, `arXiv:2501.00663`) and Sun et al. (2024, *TTT*, `arXiv:2407.04620`) translated this into differentiable sequence layers where test-time fast weights update along the gradient of momentary associative surprise $\|v_t - M_{t-1} k_t\|_2^2$ balanced by adaptive weight decay.
3. **Dual Failure Modes of Continual Learning:** Dohare, Mahmood, & Sutton (2024, *Nature*) proved that continual gradient descent suffers not only from *catastrophic forgetting* of old tasks, but also progressive **loss of plasticity** on new tasks due to weight-norm inflation, dead units, and representation rank collapse. Sustaining open-ended lifelong learning requires pairing fast/slow memory separation + experience replay with active plasticity preservation (weight-decay / shrink-and-perturb / dormant-unit regeneration).
4. **Global Workspace & Metacognitive Monitoring:** Global Neuronal Workspace theory (Baars, 1988; Dehaene et al., 2017) posits a shared, bounded-capacity latent workspace where specialized perceptual, memory, and world-model modules broadcast representations, monitored by prefrontal uncertainty/conflict signals that trigger deliberate multi-step System-2 iteration when System-1 confidence is low.

---

## 4. World Models, Causality, Planning, and Self-Improvement

- **Representation-Space World Models & Imagination Planning:** Hafner et al. (2025, *Nature*, DreamerV3) and Assran, LeCun et al. (2025, *V-JEPA 2*, `arXiv:2506.09985`) established that predicting future states in **normalized/regularized latent space** ($z_{t+1} = \text{Predictor}(z_t, a_t)$) rather than raw surface tokens avoids generative noise memorization and enables decision-time trajectory lookahead (MPC / tree search).
- **Causal vs. Observational Inference:** By Pearl's Causal Hierarchy Theorem and Schölkopf et al. (2021, *Proc. IEEE*), observational sequence predictors learn $P(Y \mid X)$ and systematically fail under confounding ($Z \to X, Z \to Y$) when asked for interventional outcomes $P(Y \mid do(X=x))$, which require graph surgery (severing incoming mechanisms into $X$ while holding independent mechanisms $P(Y \mid X, Z)$ invariant).
- **Empirical Evolutionary Self-Improvement:** Zhang, Clune et al. (ICLR 2026, *Darwin Gödel Machine*, `arXiv:2505.22954`) showed that while formal Gödel-machine proofs of beneficial self-modification are intractable, maintaining a **versioned branching archive** of architectural/algorithmic variants scored on strict held-out benchmarks enables safe, non-collapsing iterative improvement.

---

## 5. Summary Distinction: Established Findings vs. Open Speculation

| Claim / Principle | Epistemic Status | Primary Evidence |
|---|---|---|
| Fixed-depth Transformers & scalar SSMs cannot track $S_5$ state asymptotically ($\mathsf{TC}^0$) | **Established Theorem** | Merrill et al. (ICML 2024) |
| Gated Delta-Rule + Attention hybrids outperform pure linear RNNs on retrieval | **Established Empirical** | Yang et al. (ICLR 2025); Qwen3.5 |
| Standard GD loses plasticity over long continual task sequences without diversity/shrinkage | **Established Empirical** | Dohare et al. (*Nature* 2024) |
| Input-anchored latent recursive refinement scales reasoning at tiny parameter counts (7M) | **Established Empirical** | Geiping et al. (2025); Jolicoeur-Martineau (2025) |
| Complex multi-frequency brain hierarchies are necessary for small-model ARC reasoning | **Refuted / Unproven** | ARC Prize HRM Ablation (2025); TRM (2025) |
| Scaling next-token LLM parameters + internet text alone converges to AGI | **Unsupported Speculation** | Violates causal identifiability, $\mathsf{TC}^0$ depth per token, and continual plasticity |
| Unifying surprise-gated fast-weight episodic memory + input-anchored iterative latent world-model + calibrated halting/verification yields synergistic cross-domain generalization | **Testable Core Hypothesis of This Lab** | Formulated in Phase 4 & tested in Phases 6-7 |
