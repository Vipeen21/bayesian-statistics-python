
# Chapter 1: Bayes’ Theorem & Probability Foundations

This chapter establishes the theoretical foundations of Bayesian inference. We begin by defining the mathematical mechanisms of probability updates before constructing the formal architecture of Bayes' theorem.

---

## 1.1 Mathematical Architecture of Bayesian Inference

Bayes' theorem provides a rigorous axiomatic framework for updating a prior probability distribution in light of empirical evidence. Within the context of statistical inference, let $\theta \in \Theta$ denote the parameter space of interest (or the underlying hypothesis), and let $D$ represent the observed data vector sample space.

The conditional probability mapping is formally defined as:

$$P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}$$

### Structural Components of the Theorem

| Component | Mathematical Notation | Functional Definition |
| --- | --- | --- |
| **Posterior Probability** | $P(\theta \| D$ | The conditional probability distribution of the parameter $\theta$ updated after conditioning on the observed empirical data $D$. |
| **Likelihood Function** | $P(D \| \theta$ | The probability of observing data $D$ given a specific parameter configuration $\theta$. This functions as the objective data-generating mechanism. |
| **Prior Probability** | $P(\theta)$ | The marginal probability distribution expressing state-of-knowledge constraints or beliefs regarding $\theta$ prior to data observation. |
| **Marginal Likelihood (Evidence)** | $P(D)$ | The normalizing constant calculated by integrating out the parameter space: $P(D) = \int_{\Theta} P(D \| \theta) P(\theta) d\thet$. |

> **Axiomatic Interpretation:**
> The posterior distribution acts as a compromise between the historical bounds set by the prior distribution and the empirical realities captured by the likelihood function.

---

## 1.2 Parametric Paradigm: The Beta-Binomial Conjugate Model

To analyze the operational mechanics of this inference engine, we examine the problem of estimating the latent bias parameter $p$ of a Bernoulli data-generating process (e.g., sequential coin tosses).

### 1. Data-Generating Process & Likelihood

Given a sequence of $n$ independent and identically distributed (i.i.d.) Bernoulli trials resulting in $k$ successes, the joint probability mass function defines the Binomial likelihood:

$$P(D | p) = \binom{n}{k} p^k (1-p)^{n-k}$$

For an empirical sample where $n = 10$ and $k = 7$:

$$P(D | p) = \binom{10}{7} p^7 (1-p)^3$$

### 2. Prior Specification

To model our prior state of knowledge regarding $p \in [0,1]$, we utilize the Beta distribution family. A weakly informative prior is specified using hyperparameters $\alpha_0 = 2$ and $\beta_0 = 2$:

$$P(p) = \frac{1}{\text{B}(\alpha_0, \beta_0)} p^{\alpha_0 - 1} (1-p)^{\beta_0 - 1} = \frac{1}{\text{B}(2, 2)} p^1 (1-p)^1$$

### 3. Posterior Derivation via Conjugacy

Because the Beta distribution is the conjugate prior for a Binomial likelihood, the posterior distribution belongs to the same parametric family. Dropping constants independent of $p$ yields the unnormalized posterior kernel:

$$P(p | D) \propto P(D | p) \times P(p)$$

$$P(p | D) \propto \left[ p^7 (1-p)^3 \right] \times \left[ p^1 (1-p)^1 \right]$$

$$P(p | D) \propto p^{8} (1-p)^{4}$$

This kernel uniquely identifies a revised Beta distribution with updated hyperparameter dimensions:

$$\alpha_{\text{post}} = \alpha_0 + k = 2 + 7 = 9$$

$$\beta_{\text{post}} = \beta_0 + (n - k) = 2 + 3 = 5$$

Thus, the exact analytical posterior distribution is $\text{Beta}(9, 5)$.

---

## 1.3 Computational Simulation: Deterministic Grid Approximation

When analytical solutions are intractable, numerical approximations must be employed. The script below implements a deterministic grid approximation over a discrete parameter space to map out the density functions without relying on probabilistic programming sampling libraries.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# 1. Parameter Space Discretization
# Constructing a uniform grid over the bounded domain p in [0, 1]
GRID_RESOLUTION = 1000
p_grid = np.linspace(0, 1, GRID_RESOLUTION)

# 2. Prior Density Initialization
# Assuming a uniform distribution across the grid coordinates, representing a Beta(1,1) boundary
prior = np.repeat(1.0, GRID_RESOLUTION)

# 3. Likelihood Evaluation
# Evaluating the Binomial PMF for k=7 successes out of n=10 experimental trials
observed_successes = 7
total_trials = 10
likelihood = binom.pmf(k=observed_successes, n=total_trials, p=p_grid)

# 4. Posterior Calculation
# Computing the unnormalized joint distribution via element-wise multiplication
unnormalized_posterior = likelihood * prior

# 5. Numerical Integration (Normalization)
# Applying Riemann sums to ensure the total mass integrates to unity
posterior = unnormalized_posterior / np.sum(unnormalized_posterior)

# 6. Visualization of the Posterior Vector Space
plt.figure(figsize=(9, 5))
plt.plot(p_grid, posterior, label="Posterior Density $P(p|D)$", color="#1f4e79", linewidth=2)
plt.fill_between(p_grid, 0, posterior, color="#1f4e79", alpha=0.15)
plt.xlabel("Latent Success Parameter ($p$)", fontsize=11)
plt.ylabel("Probability Density", fontsize=11)
plt.title("Deterministic Grid Approximation of the Posterior Distribution", fontsize=12, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True)
plt.show()

```
