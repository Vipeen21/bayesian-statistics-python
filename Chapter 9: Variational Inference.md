# Chapter 9: Variational Inference

While Markov Chain Monte Carlo (MCMC) algorithms provide an asymptotic guarantee of converging to the true target posterior distribution $P(\boldsymbol{\theta} \mid D)$, their sampling-based nature introduces severe computational bottlenecks when applied to massive datasets or high-dimensional parameter spaces. This chapter introduces **Variational Inference (VI)**, an alternative paradigm that reframes the integration problem of Bayesian inference as a deterministic optimization problem, offering substantial computational acceleration.

---

## 9.1 Theoretical Foundations of Variational Optimization

The core objective of Variational Inference is to approximate an intractable true posterior distribution $P(\boldsymbol{\theta} \mid D)$ using a family of tractable distributions, denoted as $q(\boldsymbol{\theta} \in \mathcal{Q})$. We seek the specific distribution $q^*(\boldsymbol{\theta})$ within this family that minimizes the divergence from the true target posterior space.

### 1. The Kullback-Leibler (KL) Divergence

To operationalize the concept of "closeness" between distributions, VI utilizes the asymmetric Kullback-Leibler divergence from information theory. Mathematically, the divergence from $q(\boldsymbol{\theta})$ to $P(\boldsymbol{\theta} \mid D)$ is defined as:

$$\text{KL}\left(q(\boldsymbol{\theta}) \parallel P(\boldsymbol{\theta} \mid D)\right) = \int_{\Theta} q(\boldsymbol{\theta}) \log \left( \frac{q(\boldsymbol{\theta})}{P(\boldsymbol{\theta} \mid D)} \right) d\boldsymbol{\theta}$$

Direct minimization of this objective is impossible because the denominator contains the unknown true posterior, which requires computing the intractable marginal likelihood $P(D)$.

### 2. The Evidence Lower Bound (ELBO)

To circumvent this intractability, we use Bayes' theorem to decompose the log marginal likelihood (the log evidence), $\log P(D)$:

$$\log P(D) = \text{ELBO}(q) + \text{KL}\left(q(\boldsymbol{\theta}) \parallel P(\boldsymbol{\theta} \mid D)\right)$$

Where the **Evidence Lower Bound (ELBO)** is expressed as:

$$\text{ELBO}(q) = \mathbb{E}_{q}[\log P(D, \boldsymbol{\theta})] - \mathbb{E}_{q}[\log q(\boldsymbol{\theta})]$$

Because the log evidence $\log P(D)$ is constant with respect to $q(\boldsymbol{\theta})$, and since the KL divergence is strictly non-negative ($\text{KL} \ge 0$), **maximizing the ELBO is mathematically equivalent to minimizing the KL divergence** to the true posterior.

$$\arg\min_{q \in \mathcal{Q}} \text{KL}\left(q(\boldsymbol{\theta}) \parallel P(\boldsymbol{\theta} \mid D)\right) \equiv \arg\max_{q \in \mathcal{Q}} \text{ELBO}(q)$$

---

## 9.2 Methodological Taxonomy and Trade-offs

The architectural differences between sampling-based (MCMC) and optimization-based (VI) inference are systematic and govern their application domains.

| Evaluative Dimension | Markov Chain Monte Carlo (MCMC) | Variational Inference (VI) |
| --- | --- | --- |
| **Underlying Mechanism** | Stochastic sampling via Markov chains. | Deterministic function optimization. |
| **Asymptotic Properties** | Non-parametric; guarantees exact convergence as samples $N \to \infty$. | Bounded by the expressiveness of the chosen variational family $\mathcal{Q}$. |
| **Computational Profile** | Exceptionally slow on large datasets; scales poorly with dimensions. | Highly scalable; optimized via gradient-descent variants. |
| **Systemic Error Biases** | Stochastic variance and sampling noise. | Underestimation of posterior variance and covariance. |

### The Underestimation of Variance

A vital pedagogical caveat regarding Variational Inference is its systematic tendency to underestimate the variance of the true posterior. Because the forward KL objective $\text{KL}(q \parallel P)$ integrates over $q(\boldsymbol{\theta})$, it heavily penalizes regions where $q(\boldsymbol{\theta}) > 0$ but $P(\boldsymbol{\theta} \mid D) \to 0$. Consequently, the optimization path forces $q(\boldsymbol{\theta})$ to fit inside the mode of the true posterior, ignoring mass in heavy tails or asymmetric dimensions.

---

## 9.3 Computational Implementation: Automatic Differentiation Variational Inference

Modern probabilistic programming languages implement **Automatic Differentiation Variational Inference (ADVI)**. ADVI automatically maps bounded parameters to an unconstrained real space $\mathbb{R}^d$ and fits a spherical or full-covariance Gaussian variational distribution using stochastic gradient ascent.

The script below demonstrates the optimization execution of ADVI inside `PyMC` on the non-conjugate Gaussian model defined in Chapter 7, followed by a post-optimization validation comparing it directly against the MCMC trace bounds.

```python
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Operational Assumptions:
# 'non_conjugate_gaussian_model' is initialized as per the structural 
# specifications detailed in Section 7.3, mapping data vector 'y_observed'.

with non_conjugate_gaussian_model:
    # 1. Execute Automatic Differentiation Variational Inference (ADVI)
    # Instead of generating stochastic sample draws, we run optimization steps
    optimization_steps = 30000
    variational_approximation = pm.fit(method="advi", n=optimization_steps)

# 2. Extract Empirical Samples from the Converged Variational Object
# This allows us to map the localized optimized density function back to parameter space
vi_inferred_samples = 1000
trace_vi = variational_approximation.sample(draws=vi_inferred_samples)

# 3. Post-Estimation Comparative Diagnostic Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Plot Inferred Parametric Densities for Parameter: mu
az.plot_density(
    [trace_vi, trace_mcmc], 
    var_names=["mu"], 
    data_labels=["VI Approximation (ADVI)", "MCMC Reference (NUTS)"],
    colors=["#1f4e79", "#d9534f"],
    ax=axes[0]
)
axes[0].set_title("Posterior Density Comparison for Location Parameter $\mu$")
axes[0].set_xlabel("$\mu$")

# Plot Inferred Parametric Densities for Parameter: sigma
az.plot_density(
    [trace_vi, trace_mcmc], 
    var_names=["sigma"], 
    data_labels=["VI Approximation (ADVI)", "MCMC Reference (NUTS)"],
    colors=["#1f4e79", "#d9534f"],
    ax=axes[1]
)
axes[1].set_title("Posterior Density Comparison for Scale Parameter $\sigma$")
axes[1].set_xlabel("$\sigma$")

plt.suptitle(
    "Structural Divergence Analysis: Optimizing VI (Blue) vs. Asymptotic MCMC (Red)", 
    fontsize=12, 
    fontweight="bold"
)
plt.tight_layout()
plt.show()

```
