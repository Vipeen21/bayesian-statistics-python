# Chapter 14: Prior Elicitation and Sensitivity Analysis

The specification of the prior distribution $P(\theta)$ represents a foundational step in building a coherent Bayesian model. Because prior selection incorporates existing knowledge before data assimilation, we must verify that our statistical conclusions are robust and not overly sensitive to arbitrary assumptions. This chapter establishes formal methodologies for translating structural knowledge into mathematical distributions (prior elicitation) and outlines strategies for quantifying prior influence (sensitivity analysis).

---

## 14.1 Epistemological Foundations of Prior Elicitation

Prior elicitation is the formal process of translating expert domain knowledge, physical constraints, or historical empirical records into well-defined probability distributions. Rather than selecting mathematical priors solely for computational convenience, we structure them based on their information content:

* **Informative Priors:** Formulated when historical data, mathematical proofs, or physical laws restrict the parameter space $\Theta$ to localized fields. These priors possess low variance and exert a noticeable influence on the resulting posterior density.
* **Weakly Informative Priors:** Explicitly designed to regularize the parameter space. They supply enough structural bounds to prevent computationally unstable or physically impossible parameter values (e.g., negative variance) while allowing the likelihood function to dominate the posterior distribution.
* **Non-Informative (Diffuse) Priors:** Structured to assign equal or near-equal probability weight across the parameter domain (e.g., a wide uniform distribution $\mathcal{U}(-1000, 1000)$). While intended to reflect complete initial ignorance, they can introduce severe geometric pathologies into complex multi-dimensional samplers and lack invariance under mathematical parameter transformations.

---

## 14.2 Mathematical Theory of Posterior Sensitivity

A **Prior Sensitivity Analysis** systematically evaluates the stability of the posterior distribution $P(\theta \mid D)$ by changing the underlying prior specification across a spectrum of competing hypotheses.

Let $\mathcal{P} = \{P_1(\theta), P_2(\theta), \dots, P_M(\theta)\}$ define a set of candidate prior distributions representing different initial states of knowledge. The model undergoes estimation across each prior variant while keeping the data likelihood $P(D \mid \theta)$ strictly constant:

$$P_m(\theta \mid D) = \frac{P(D \mid \theta) P_m(\theta)}{\int_{\Theta} P(D \mid \theta) P_m(\theta) d\theta}$$

### The Asymptotic Convergence Metric

According to the **Bernstein-von Mises Theorem**, under standard regularity conditions, the posterior distribution converges asymptotically to a Normal distribution centered at the true parameter value $\theta_0$ as the sample size approaches infinity ($n \to \infty$), regardless of the initial prior configuration:

$$\lim_{n \to \infty} \left\| P_m(\theta \mid D) - \mathcal{N}\left(\hat{\theta}_{\text{MLE}}, \, \frac{1}{I_n(\theta_0)}\right) \right\| = 0$$

Where $I_n(\theta_0)$ represents the Fisher information matrix. However, in small-sample regimes ($n \ll \infty$), the choices made during prior elicitation can significantly alter the posterior mean, variance, and credible intervals. A sensitivity analysis maps these variations to quantify the epistemic stability of our statistical inferences.

---

## 14.3 Computational Implementation: Quantifying Prior Influence

The program below evaluates prior sensitivity within a Bernoulli data-generating process using the standard coin-tossing model ($n=10, k=7$). We compare a weakly informative regularizing prior against a highly informative, dogmatic prior to observe how different hyperparameter boundaries affect the final posterior distribution.

```python
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# 1. Empirical Observation Matrix (7 Successes within 10 Independent Trials)
observed_successes = 7
total_trials = 10

# 2. Model 1 Specification: Weakly Informative Beta(2, 2) Prior
with pm.Model() as weak_prior_model:
    # A prior representing minor regularizing bounds centered at 0.5
    p = pm.Beta("p", alpha=2.0, beta=2.0)
    y = pm.Binomial("y", n=total_trials, p=p, observed=observed_successes)
    
    print("[System Notice] Executing MCMC Chain for Weakly Informative Framework...")
    trace_weak = pm.sample(draws=1000, tune=500, return_inferencedata=True, random_seed=42)

# 3. Model 2 Specification: Strongly Informative Beta(50, 50) Prior
with pm.Model() as strong_prior_model:
    # A rigid prior reflecting a strong initial belief that the system is perfectly fair
    p = pm.Beta("p", alpha=50.0, beta=50.0)
    y = pm.Binomial("y", n=total_trials, p=p, observed=observed_successes)
    
    print("[System Notice] Executing MCMC Chain for Strongly Informative Framework...")
    trace_strong = pm.sample(draws=1000, tune=500, return_inferencedata=True, random_seed=42)

# 4. Comprehensive Post-Sampling Sensitivity Visualization
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), sharex=True)

# Plotting the posterior distribution conditioned on the weak prior
az.plot_posterior(trace_weak, var_names=["p"], color="#1f4e79", hdi_prob=0.94, ax=axes[0])
axes[0].set_title("Posterior Density under a Weak Prior: Beta(2, 2)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Parameter Space ($p$)")

# Plotting the posterior distribution conditioned on the strong prior
az.plot_posterior(trace_strong, var_names=["p"], color="#228b22", hdi_prob=0.94, ax=axes[1])
axes[1].set_title("Posterior Density under a Dogmatic Prior: Beta(50, 50)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Parameter Space ($p$)")

plt.suptitle("Prior Sensitivity Analysis: Posterior Displacements", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

```

### Comparative Analysis of Posterior Displacements

The table below outlines the structural shifts in parameter estimation caused by the different prior assumptions:

| Metric | Weak Prior Framework $\text{Beta}(2,2)$ | Strong Prior Framework $\text{Beta}(50,50)$ | Analytical Interpretation |
| --- | --- | --- | --- |
| **Prior Central Mass** | $0.5000$ | $0.5000$ | Both prior distributions are centered symmetrically at the identical midpoint coordinate. |
| **Empirical Sample Ratio ($k/n$)** | $0.7000$ | $0.7000$ | The underlying empirical data vector remains constant across both models. |
| **Resulting Posterior Mean** | $\approx 0.6429$ | $\approx 0.5182$ | The weak prior allows the sample data to drive the posterior. The dogmatic prior resists the empirical evidence, shrinking the estimate back toward $0.5$. |
| **94% Credible Interval (HDI)** | Wide Bounds | Narrow Bounds | The high precision of the strong prior artificially restricts the posterior variance, showing that small samples cannot easily overcome highly informative prior assumptions. |
