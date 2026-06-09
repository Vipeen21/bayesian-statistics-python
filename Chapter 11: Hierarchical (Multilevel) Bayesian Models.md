# Chapter 11: Hierarchical (Multilevel) Bayesian Models

Data structures in empirical research frequently exhibit nested or grouped topologies (e.g., asset returns nested within economic sectors, or student test scores nested within individual academic institutions). Classical statistical estimation strategies typically handle grouped data through two extreme paradigms: pooling all observations into a singular global matrix, or estimating completely independent models for each group. This chapter introduces **Hierarchical Bayesian Modeling**, a framework that establishes intermediate parameters to bridge these extremes via partial pooling.

---

## 11.1 Theoretical Framework: Pooling Taxonomies

To understand the mathematical utility of hierarchical structures, we examine the three primary pooling methodologies used to estimate group-level parameters $\theta_j$ across $J$ distinct groups.

| Estimation Paradigm | Mathematical Formulation | Structural Assumption | Statistical Trade-off |
| --- | --- | --- | --- |
| **Complete Pooling** | $\theta_1 = \theta_2 = \dots = \theta_J = \mu$ | Groups are completely identical; intra-group variation is treated as random noise. | Introduces severe parameter bias if groups are heterogeneous; masks localized effects. |
| **No Pooling** | $\theta_j \sim \mathcal{N}(\mu_j, \sigma_j^2)$ | Groups are completely independent; historical data from group $j$ contains no information relevant to group $k$. | Overfits small sample sizes; results in high estimator variance for data-sparse groups. |
| **Partial Pooling** (Hierarchical) | $\theta_j \sim \mathcal{N}(\mu, \tau^2)$ | Group parameters are latent random variables drawn from a shared, higher-level population distribution. | Optimal bias-variance compromise via shrinkage; allows data-sparse groups to borrow statistical strength. |

### The Mechanics of Shrinkage

In a hierarchical model, the posterior estimate for an individual group mean $\theta_j$ is a weighted compromise between its localized sample mean $\bar{y}_j$ and the global population mean $\mu$:

$$\hat{\theta}_j \approx \frac{\frac{n_j}{\sigma^2}}{\frac{n_j}{\sigma^2} + \frac{1}{\tau^2}}\bar{y}_j + \frac{\frac{1}{\tau^2}}{\frac{n_j}{\sigma^2} + \frac{1}{\tau^2}}\mu$$

Where $n_j$ is the sample size of group $j$, $\sigma^2$ is the data-level variance, and $\tau^2$ is the between-group variance parameter. As $n_j \to 0$ or $\tau \to 0$, the estimate shrinks toward the global average $\mu$. Conversely, as $n_j \to \infty$ or $\tau \to \infty$, the estimate moves toward the unpooled local sample mean $\bar{y}_j$.

---

## 11.2 The Canonical "Eight Schools" Model Structure

The classic "Eight Schools" dataset examine the effects of special coaching programs on SAT V scores across eight independent secondary schools ($J = 8$). Let $y_j$ denote the observed estimated treatment effect in school $j$, and let $\sigma_j$ represent the associated standard error of measurement, treated as known.

The structural mathematical model is formulated through a multi-layered probability hierarchy:

### Layer 1: Data Likelihood

$$y_j \sim \mathcal{N}(\theta_j, \sigma_j^2), \quad \text{for } j = 1, \dots, 8$$

### Layer 2: Population Hyper-Distribution

$$\theta_j \sim \mathcal{N}(\mu, \tau^2)$$

### Layer 3: Hyperprior Specifications

$$\mu \sim \mathcal{N}(0, 5^2), \quad \tau \sim \text{Half-Cauchy}(\beta = 5)$$

Where $\mu$ represents the global average coaching effect, and $\tau$ represents the standard deviation of treatment effects across schools.

---

## 11.3 Computational Implementation: Non-Centered Parameterization

When the between-group variance parameter $\tau$ approaches zero, the geometry of a centered hierarchical model forms a sharp, high-curvature topology known as **Rubin’s Funbler**. Standard Hamiltonian Monte Carlo (HMC) transitions fail in these configurations, resulting in divergent transitions.

To ensure stable sampling, we apply a **Non-Centered Parameterization**, shifting the dependency on $\tau$ out of the prior distribution parameter arguments and into a deterministic equation layer:

$$\theta_j = \mu + \tau \cdot \theta_{\text{offset}, j}, \quad \theta_{\text{offset}, j} \sim \mathcal{N}(0, 1)$$

The script below implements this model architecture within `PyMC`.

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# 1. Empirical Structural Data Formulation
J = 8
y_schools = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma_schools = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

# 2. Hierarchical Probabilistic Model Configuration
with pm.Model() as hierarchical_model:
    # Top-Level Hyperpriors
    mu = pm.Normal("mu", mu=0, sigma=5)
    tau = pm.HalfCauchy("tau", beta=5)
    
    # Non-Centered Parameterization Layer to Prevent Funnel Divergences
    theta_offset = pm.Normal("theta_offset", mu=0, sigma=1, shape=J)
    
    # Deterministic Realization of the Localized Parameters
    theta = pm.Deterministic("theta", mu + tau * theta_offset)
    
    # Group-Level Likelihood Vector Model
    obs = pm.Normal("obs", mu=theta, sigma=sigma_schools, observed=y_schools)
    
    # 3. Execution of the Gradient-Based MCMC Sampler
    trace_hierarchical = pm.sample(
        draws=2000, 
        tune=2000, 
        target_accept=0.95, 
        return_inferencedata=True, 
        random_seed=42
    )

# 4. Post-Estimation Visual Diagnostics and Verification
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_forest(
    trace_hierarchical, 
    var_names=["theta"], 
    combined=True, 
    r_hat=True, 
    ax=ax
)
ax.set_title("Posterior Quantile Forest Plot: Borrowing Strength via Partial Pooling", fontsize=12, fontweight="bold")
ax.set_xlabel("Estimated Coaching Effect ($\theta_j$)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()

```

### Analysis of the Forest Plot Profile

The generated forest plot visualizes the posterior credible intervals for each school's underlying effect ($\theta_j$). Notice that the point estimates are pulled systematically away from their raw sample averages ($y_{\text{schools}}$) toward the global estimated mean $\mu$. This shrinkage effect is most pronounced for schools with high measuring uncertainty (large $\sigma_j$ entries, such as School 8), demonstrating how hierarchical models mitigate data noise through partial pooling.
