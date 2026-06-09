# Chapter 7: Markov Chain Monte Carlo (MCMC) – Part I

For the vast majority of high-dimensional parametric models encountered in empirical research, the marginal likelihood integration in the denominator of Bayes' theorem is analytically intractable. This structural limitation necessitates the deployment of Markov Chain Monte Carlo (MCMC) algorithms—stochastic simulation techniques designed to sample from the joint posterior distribution vector space without computing the normalizing constant.

---

## 7.1 The Metropolis-Hastings Algorithm

The Metropolis-Hastings (MH) algorithm forms the foundational benchmark for stochastically exploring complex probability spaces. It constructs an ergodic Markov chain whose stationary distribution matches the target joint posterior density exactly.

### Mathematical Mechanics of the MH Sequence

Let $\theta^{(t)}$ denote the current state of the parameter vector in a continuous space $\Theta$. The algorithm transitions to state $\theta^{(t+1)}$ via a two-stage stochastic optimization pipeline:

1. **Proposal Phase:** A candidate state $`\theta^*`$
 is drawn from a user-defined proposal distribution (or transition kernel) $`q(\theta^* \mid \theta^{(t)})`$.
2. **Acceptance-Rejection Phase:** The candidate state is evaluated against the localized density volume of the current state. The probability of transition acceptance, $\alpha(\theta^{(t)}, \theta^*)$, is computed precisely as:

```math
\alpha(\theta^{(t)}, \theta^*) = \min\left(1, \frac{P(D \mid \theta^*) P(\theta^*) q(\theta^{(t)} \mid \theta^*)}{P(D \mid \theta^{(t)}) P(\theta^{(t)}) q(\theta^* \mid \theta^{(t)})}\right) 
```



If the proposal distribution is symmetric, such that $q(\theta^* \mid \theta^{(t)}) = q(\theta^{(t)} \mid \theta^*)$ (e.g., a Gaussian random walk), the expression reduces to the standard Metropolis Hastings ratio:


```math
\alpha(\theta^{(t)}, \theta^*) = \min\left(1, \, \frac{P(D \mid \theta^*) P(\theta^*)}{P(D \mid \theta^{(t)}) P(\theta^{(t)})}\right) = \min\left(1, \, \frac{\text{Unnormalized Posterior}(\theta^*)}{\text{Unnormalized Posterior}(\theta^{(t)})}\right)
```

A random uniform variable $u \sim \mathcal{U}(0, 1)$ is sampled. The next step in the Markov chain is determined by the decision rule:

```math
\theta^{(t+1)} = \begin{cases} \theta^* & \text{if } u \le \alpha(\theta^{(t)}, \theta^*) \\ \theta^{(t)} & \text{if } u > \alpha(\theta^{(t)}, \theta^*) \end{cases}
```

---

## 7.2 Gibbs Sampling

Gibbs sampling is a specialized, structurally efficient variant of the Metropolis-Hastings framework tailored for multi-dimensional parameter spaces where localized full conditional distributions can be analytically derived.

Instead of proposing a simultaneous multi-axis shift across the entire parameter vector $\boldsymbol{\theta} = (\theta_1, \theta_2, \dots, \theta_d)'$, Gibbs sampling breaks the task down into sequential, uni-variate steps. Each coordinate is systematically updated by conditioning on the most up-to-date values of all other parameters in the network.

### The Cyclic Update Architecture

Given a $d$-dimensional parameter state vector at iteration $t$, the transition to iteration $t+1$ executes through a systematic loop of conditional extractions:

```math
\theta_1^{(t+1)} &\sim P(\theta_1 \mid \theta_2^{(t)}, \theta_3^{(t)}, \dots, \theta_d^{(t)}, D) \\
\theta_2^{(t+1)} &\sim P(\theta_2 \mid \theta_1^{(t+1)}, \theta_3^{(t)}, \dots, \theta_d^{(t)}, D) \\
\theta_3^{(t+1)} &\sim P(\theta_3 \mid \theta_1^{(t+1)}, \theta_2^{(t+1)}, \dots, \theta_d^{(t)}, D) \\
&\;\vdots \\
\theta_d^{(t+1)} &\sim P(\theta_d \mid \theta_1^{(t+1)}, \theta_2^{(t+1)}, \dots, \theta_{d-1}^{(t+1)}, D)
\end{aligned}
```

> **Operational Property of Gibbs Trajectories:**
> Because the proposal steps are drawn directly from the exact full conditional distributions, the acceptance probability $\alpha$ evaluates identically to 1 at every step. This makes Gibbs sampling highly efficient for models with conjugate sub-structures, though it remains prone to slow convergence when parameters exhibit high posterior correlation.

---

## 7.3 Computational Implementation: Analysis of Non-Conjugate Systems

We demonstrate MCMC sampling by estimating a model where the data follow a Normal distribution with both the mean $\mu$ and standard deviation $\sigma$ treated as unknown. Because the joint prior-likelihood space lacks simple conjugacy, numerical estimation via probabilistic programming is required.

The script below uses `PyMC` to initialize an advanced No-U-Turn Sampler (NUTS)—a modern Hamiltonian Monte Carlo approach that extends basic Metropolis-Hastings by utilizing gradient information to sample efficiently from complex multi-dimensional spaces.

```python
import pymc as pm
import numpy as np
import arviz as az

# 1. Empirical Synthetic Data-Generating Process
np.random.seed(42)
true_mean = 10.0
true_standard_deviation = 2.0
sample_size = 100

empirical_observations = np.random.normal(
    loc=true_mean, 
    scale=true_standard_deviation, 
    size=sample_size
)

# 2. Probabilistic Model Architectural Configuration
with pm.Model() as non_conjugate_gaussian_model:
    
    # Prior Specification for the Location Vector Space
    mu = pm.Normal("mu", mu=0.0, sigma=10.0)
    
    # Scale Parametrization Constrained to Positivity Bounds
    sigma = pm.HalfNormal("sigma", sigma=5.0)
    
    # Likelihood Model Mapping the Continuous Normal Density Function
    y_observed = pm.Normal(
        "y_obs", 
        mu=mu, 
        sigma=sigma, 
        observed=empirical_observations
    )
    
    # 3. Execution of the MCMC Sampling Routine (Gradient-Based NUTS)
    inference_data = pm.sample(
        draws=2000, 
        tune=1000, 
        return_inferencedata=True, 
        random_seed=42
    )

# 4. Rigorous Post-Sampling Diagnostic Evaluations
print("--- MCMC Convergence Diagnostics & Summary Profiles ---")
statistical_summaries = az.summary(inference_data, var_names=["mu", "sigma"])
print(statistical_summaries)

```
