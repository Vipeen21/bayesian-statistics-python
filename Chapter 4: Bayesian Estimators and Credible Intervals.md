# Chapter 4: Bayesian Estimators and Credible Intervals

While the joint posterior distribution $P(\theta \mid D)$ encapsulates the complete, mathematically rigorous state of knowledge regarding a parameter space, practical statistical application requires functional summaries. This chapter formalizes the reduction of posterior densities into optimal point estimates and probabilistic interval regions.

---

## 4.1 Theory of Bayesian Point Estimation

Bayesian point estimation treats parameter selection as an optimization problem over a localized loss domain. Rather than relying on asymptotic behavior, we extract central tendency measures directly from the conditioned probability density function.


### Mathematical Foundations of Classical Point Summaries

| Estimator Class | Mathematical Formulation | Optimization Characterization |
| :--- | :--- | :--- |
| **Posterior Mean** | $\hat{\theta}_{\text{Mean}}=\mathbb{E}[\theta\mid D]=\int_{\Theta}\theta P(\theta\mid D)d\theta$ | Minimizes expected quadratic loss: $L(\theta,\hat{\theta})=(\theta-\hat{\theta})^2$. Extremely sensitive to heavy tails. |
| **Posterior Median** | $\hat{\theta}_{\text{Median}}\implies\int_{-\infty}^{\hat{\theta}}P(\theta\mid D)d\theta=0.5$ | Minimizes expected absolute error loss: $L(\theta,\hat{\theta})=\lvert\theta-\hat{\theta}\rvert$. Highly robust to skewness. |
| **Maximum A Posteriori (MAP)** | $\hat{\theta}_{\text{MAP}}=\arg\max_{\theta}P(\theta\mid D)=\arg\max_{\theta}P(D\mid\theta)P(\theta)$ | Minimizes a 0-1 loss framework. Represents the localized mode; ignores overall distribution volume and geometry. |

> **Structural Note on the MAP Estimator:**
> The MAP acts as the Bayesian analog to the Frequentist Maximum Likelihood Estimate (MLE), augmented by prior regularizing constraints. However, because it ignores the broader geometric mass of the distribution, it can prove highly unrepresentative in complex, multi-modal, or highly asymmetrical parameter spaces.

---

## 4.2 Mathematical Formulations of Credible Regions

A Bayesian credible interval is a true probability space defined over an integrated subset of the posterior distribution. Given an alpha level $\alpha \in (0, 1)$, a $(1 - \alpha)\%$ credible region for $\theta$ is a subset $C \subset \Theta$ such that:

$$P(\theta \in C \mid D) = \int_{C} P(\theta \mid D) \, d\theta = 1 - \alpha$$

We distinguish between two structural topologies for constructing these intervals:

### 1. Equal-Tailed Intervals (ETI)

An Equal-Tailed Interval isolates the target probability area by placing equal probability mass $\alpha / 2$ into both the lower and upper bounds of the distribution tails. If $F^{-1}(\cdot)$ represents the inverse cumulative distribution function (quantile function) of the posterior, the interval limits $[\mathbb{L}, \mathbb{U}]$ are explicitly defined by:

$$\mathbb{L} = F^{-1}\left(\frac{\alpha}{2}\right), \quad \mathbb{U} = F^{-1}\left(1 - \frac{\alpha}{2}\right)$$

While computationally trivial to extract via standard empirical percentiles, ETIs are structurally sub-optimal for asymmetric posterior distributions, as they can include parameter values with lower probability densities than points excluded outside the interval bounds.

### 2. Highest Posterior Density (HPD) Intervals

The Highest Posterior Density interval guarantees optimal structural compactness. An HPD region $C_{\text{HPD}}$ is uniquely defined by a threshold value $k_\alpha$ such that:

$$C_{\text{HPD}} = \{ \theta \in \Theta : P(\theta \mid D) \ge k_\alpha \}$$

subject to the integration constraint:

$$\int_{C_{\text{HPD}}} P(\theta \mid D) \, d\theta = 1 - \alpha$$

> **Axiomatic Properties of HPD Intervals:**
> 1. **Minimal Volume:** For any given probability mass $1-\alpha$, the HPD interval occupies the smallest possible hyper-volume within the parameter space $\Theta$.
> 2. **Density Dominance:** Every point inside the HPD interval possesses a strictly higher posterior probability density than any point exterior to it. For highly skewed or multi-modal distributions, the HPD can manifest as disjoint, non-contiguous intervals.
> 
> 

---

## 4.3 Computational Implementation: Posterior Extrapolations

The computational pipeline below demonstrates how to programmatically extract point summaries and optimal Highest Density Intervals (HDI) from simulated Markov Chain Monte Carlo (MCMC) sampling data frameworks using the `ArViZ` diagnostic suite.

```python
import numpy as np
import arviz as az

# Operational Assumptions: 
# The variable 'trace' represents a valid InferenceData object generated 
# by an upstream MCMC sampler mapping the latent success parameter 'p'.

# 1. Generate Comprehensive Statistical Summary Matrices
summary_dataframe = az.summary(trace, var_names=["p"])
print("--- Structural Posterior Summaries ---")
print(summary_dataframe)

# 2. Extract Exact Center-Mass Location Parameters
posterior_mean_estimate = summary_dataframe["mean"].values[0]
posterior_median_estimate = summary_dataframe["median"].values[0]

# 3. Extract High-Density Region Vector Bounds
# By default, ArViZ computes the Highest Density Interval (HDI) at the 94% threshold
hdi_bounds = summary_dataframe[["hdi_3%", "hdi_97%"]].values[0]

# 4. Formally Format Statistical Estimators
print("\n--- Point Estimation Metrics ---")
print(f"Calculated Posterior Mean (E[p|D]):        {posterior_mean_estimate:.4f}")
print(f"Calculated Posterior Median:              {posterior_median_estimate:.4f}")

print("\n--- Optimal Interval Topologies ---")
print(f"94% Highest Density Interval Bounds (HDI): [{hdi_bounds[0]:.4f}, {hdi_bounds[1]:.4f}]")

```
