# Chapter 3: Conjugate Priors & Posterior Distributions

In our preliminary analysis of the Beta-Binomial model, we observed that combining a Beta prior distribution with a Binomial likelihood function yields a posterior distribution that remains within the Beta family. This algebraically tractability is a manifestation of mathematical conjugacy. This chapter formalizes the algebraic mechanics of conjugate families and demonstrates their implementation within computational probabilistic frameworks.

---

## 3.1 The Mathematical Theory of Conjugacy

Let $\mathcal{F}$ denote a class of probability distributions over a parameter space $\Theta$. A prior distribution $P(\theta) \in \mathcal{F}$ is defined as **conjugate** to a given likelihood function $P(D \mid \theta)$ if the resulting conditional posterior distribution $P(\theta \mid D)$ also belongs to the class $\mathcal{F}$ for all possible data outcomes $D$.

Formally, let $\mathcal{P}$ represent the family of prior distributions, and let $\mathcal{L}$ denote the parametric family of the likelihood. Conjugacy implies a closed algebraic mapping:

```math
\left\{ P(\theta) \in \mathcal{P} \quad \wedge \quad P(D \mid \theta) \in \mathcal{L} \right\} \implies P(\theta \mid D) \in \mathcal{P}
```

### Analytical and Pedagogical Significance

* **Exact Analytical Tractability:** The integration required to compute the marginal likelihood (evidence) in the denominator of Bayes' theorem, $P(D) = \int_{\Theta} P(D \mid \theta)P(\theta)d\theta$, can be evaluated analytically, bypassing computationally intensive numerical methods.
* **Deterministic Hyperparameter Updates:** The inferential pipeline reduces to a deterministic algebraic system. The posterior parameters are updated by adding empirical summary statistics directly to the prior hyperparameters.

---

## 3.2 Canonical Conjugate Parametric Families

The table below catalogs the fundamental conjugate pairs utilizing standard statistical distribution families alongside their closed-form hyperparameter update mechanisms.

Observational Likelihood Model P(D∣θ),Target Latent Parameter (θ),Assigned Conjugate Prior P(θ),Functional Hyperparameter Update Rule (P(θ∣D))
"BinomialB(n,p)",p (Success Probability),"BetaBeta(α,β)",αpost​=α+∑i=1n​xi​βpost​=β+n−∑i=1n​xi​
PoissonPoi(λ),λ (Rate Parameter),"GammaGamma(α,β)",αpost​=α+∑i=1n​xi​βpost​=β+n
"Normal (Known Variance σ2)N(μ,σ2)",μ (Population Mean),"NormalN(μ0​,σ02​)",μpost​=(σ02​μ0​​+σ2∑xi​​)⋅σpost2​σpost2​=(σ02​1​+σ2n​)−1
"MultinomialM(n,p)",p (Probability Vector),DirichletDirichlet(α),αpost​=α+x

---

## 3.3 Computational Simulation: Verification via Probabilistic Programming

While conjugate models yield exact analytical solutions, verifying these outcomes via Markov Chain Monte Carlo (MCMC) sampling techniques provides a structural baseline for analyzing complex, non-conjugate models in later phases.

The implementation below uses `PyMC` to empirically simulate the posterior vector space of the Beta-Binomial model analyzed in Chapter 1 ($n=10, k=7$), verifying convergence toward the theoretical analytical distribution $\text{Beta}(9, 5)$.

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import beta

# 1. Theoretical Parameter Definitions (Derived Analytically)
alpha_prior, beta_prior = 2, 2
observed_successes = 7
observed_failures = 3

alpha_post = alpha_prior + observed_successes
beta_post = beta_prior + observed_failures

# 2. Probabilistic Model Specification
with pm.Model() as conjugate_coin_model:
    # Prior Distribution Elicitation: p ~ Beta(2, 2)
    p = pm.Beta("p", alpha=alpha_prior, beta=beta_prior)
    
    # Likelihood Definition: Likelihood ~ Binomial(n=10, p)
    likelihood = pm.Binomial(
        "lik", 
        n=observed_successes + observed_failures, 
        p=p, 
        observed=observed_successes
    )
    
    # 3. Execution of Hamiltonian Monte Carlo (NUTS Sampler)
    # Drawing samples to map out the target density empirically
    inference_data = pm.sample(
        draws=2000, 
        tune=1000, 
        return_inferencedata=True, 
        random_seed=42
    )

# 4. Post-Sampling Statistical Verification & Visualization
fig, ax = plt.subplots(figsize=(9, 5))

# Plot empirical MCMC posterior trace distribution
az.plot_posterior(
    inference_data, 
    var_names=["p"], 
    kind="kde", 
    ax=ax, 
    label="Empirical MCMC Posterior"
)

# Overlay theoretical exact conjugate distribution
p_axis = np.linspace(0, 1, 500)
theoretical_density = beta.pdf(p_axis, alpha_post, beta_post)

# Rescale theoretical density to line up with normalized KDE coordinates
# PyMC plot_posterior visualizes standardized densities
ax.plot(
    p_axis, 
    theoretical_density / np.max(theoretical_density) * 0.9, 
    color="#d9534f", 
    linestyle="--", 
    linewidth=2, 
    label=f"Analytical Exact Bound: Beta({alpha_post}, {beta_post})"
)

ax.set_xlabel("Parameter Space Vector ($p$)", fontsize=11)
ax.set_ylabel("Normalized Density Scale", fontsize=11)
ax.set_title("Empirical MCMC Trace vs. Exact Conjugate Analytical Posterior", fontsize=12, fontweight="bold")
ax.legend(loc="upper left", frameon=True)
ax.grid(True, linestyle=":", alpha=0.6)
plt.show()

```
