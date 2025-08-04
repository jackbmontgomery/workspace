# Workspace

Collection of single file implementations from various areas of interest.

## `rk.py`

This module provides an implementation of adaptive step size Runge-Kutta methods, specifically the **RK23** and **RK45** integrators. The implementation draws inspiration from `scipy.integrate`, while relying heavily on the foundational treatment of these methods in Hairer et al. [1].

One point of divergence between this implementation and `scipy` is the handling of the **E** matrix. In standard Runge-Kutta notation:

- The **C** vector specifies the time offsets (as fractions of the step size) at which intermediate function evaluations are performed.
- The **A** matrix defines the coefficients used to compute the intermediate stages (i.e., the \(k_i\) terms).
- The **B** vector provides the weights for combining these intermediate gradients to obtain the solution at the next time step, \(y_{t+1}\).

However, the role of the **E** vector/matrix is less clearly defined in some references and not explicitly discussed in Hairer.

In Hairer's treatment, the local error is estimated by comparing two solutions of different orders, typically a higher-order estimate \(y_{t+1}\) and a lower-order estimate \(\hat{y}\). The local truncation error is given by:

$$
\text{error} = |y_{t+1} - \hat{y}|
$$

where:

$$
y_{t+1} = \sum_{i=0}^{n} b_i k_i, \quad \hat{y} = \sum_{i=0}^{n} \hat{b}_i k_i
$$

This can be rewritten as:

$$
\text{error} = \left| \sum_{i=0}^{n} (b_i - \hat{b}_i) k_i \right| = \sum_{i=0}^{n} e_i k_i
$$

where \(e_i = b_i - \hat{b}_i\) represents the difference between the coefficients of the high- and low-order estimates. This \(e_i\) is often what the **E** vector represents in practice — a convenient way to express the embedded error estimator.


### Reference

[1] E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, Springer, 1993.

---

## `eqx-dqn.py`

This module provides a JAX-based implementation of Deep Q-Network (DQN) reinforcement learning using the **Equinox** framework for neural networks and **Flashbax** for experience replay. The implementation follows the foundational DQN algorithm introduced by Mnih et al. [1], with modern JAX functional programming paradigms.

### Key Components

#### Neural Network Architecture
The `DQN` class implements a simple feedforward network: input → 32 → 32 → actions, with ReLU activations. This maps state observations to Q-values Q(s,a) for each possible action.

#### Experience Replay and Target Networks
The implementation maintains:
- **Experience Buffer**: Stores transitions `(s_t, a_t, r_t, s_{t+1}, terminated_t)` using Flashbax
- **Target Network**: Provides stable Q-value targets, updated periodically via hard updates ($\tau=1$)

#### Scan-Based Training Loop
The core innovation is using `jax.lax.scan` for the training loop, which JIT-compiles the entire episodic learning process. The scan maintains a `Carry` state containing network parameters, optimizer state, environment state, and replay buffer state.

For reference, this scan based training is about 50 times faster than the same `pytorch` implementation. (0.27s vs 12.61s)

### Training Mechanics

**Action Selection**: Standard $\epsilon$-greedy policy with exponential decay: $\epsilon_t = \max(\epsilon_{t-1} \times \text{decay\_rate}, \epsilon_{\min})$

**Loss Function**: Temporal difference error following standard DQN:
$
L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q_{\text{target}}(s', a') - Q(s, a))^2]
$

**Termination Handling**: The code carefully distinguishes between episode termination (environment conditions) and truncation (time limits), which is crucial for proper bootstrapping in the Bellman equation.

### Implementation Benefits

Using `jax.lax.scan` instead of Python loops provides:
- **Compilation**: Entire training loop JIT-compiled as single function
- **Efficiency**: Eliminates Python overhead
- **Reproducibility**: Deterministic execution with proper random key management

The functional programming approach ensures immutable state updates and clean separation between pure functions and stateful operations.

### Reference
[1] V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529-533, 2015.

---
