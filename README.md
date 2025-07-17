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

---

### Reference

[1] E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, Springer, 1993.
