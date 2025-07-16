# Workspace

Collection of single file implementations from various areas of interest.

## rk.py

This is an implementation of dynamic step Runge Kutta methods. Specifically RK23 and RK45.
The implementation takes inspiration from the scipy.integrate package and uses Hairer [1] extensively.

The one point where the scipy implementation differs from Hairer is the **E** Matrix.
We know that the C matrix represents the coefficients of the step size in the time argument to the differential function.
While A contains the coefficients for the intermediate gradient samples.
And B contains the coefficients for the intermediate gradients to obtain y at the next time point.
It was not clear what the purpose of E was.

In Hairer, the local error is estimated as:
$$
\text{error} = |y_{t+1} - \hat{y}|
$$

Where:
$$
y_{t+1}= \sum^n_{i = 0} b_i k_i
$$

and
$$
\hat{y}= \sum^n_{i = 0} \hat{b}_i k_i
$$

So we can write the error as:
$$
\text{error} = |\sum^n_{i = 0} b_i k_i - \sum^n_{i = 0} \hat{b_i} k_i| = \sum^n_{i = 0} (b_i - \hat{b}_i) k_i
$$

We define $e_i = b_i - \hat{b}_i$ 

[1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations I: Nonstiff Problems"
