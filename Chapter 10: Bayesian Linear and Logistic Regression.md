# Chapter 10: Bayesian Linear and Logistic Regression

This chapter generalizes the foundations of Bayesian inference to the domain of generalized linear models (GLMs). We transition from single-parameter estimation to conditioning multi-dimensional parameter vectors on multivariate continuous and discrete data structures.

---

## 10.1 Classical Structural Foundations of Bayesian Linear Regression

In classical linear estimation, ordinary least squares (OLS) or maximum likelihood estimation (MLE) yields singular point configurations for the weight vector, conditioning uncertainty assessments on asymptotic Gaussian assumptions. The Bayesian formulation treats the regression hyperplanes as a random vector space, computing a joint posterior distribution over all parameters simultaneously.

### 1. Mathematical Model Specification

Let $y_i \in \mathbb{R}$ represent the target dependent variable for observation $i$, and let $\mathbf{x}_i \in \mathbb{R}^d$ denote the corresponding vector of explanatory variables. The structural linear data-generating process is formulated as:

$$y_i = \beta_0 + \mathbf{\beta}^T \mathbf{x}_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

This is equivalent to stating that the conditional likelihood of the data follows a Normal distribution:

$$P(y_i \mid \mathbf{x}_i, \beta_0, \mathbf{\beta}, \sigma) = \mathcal{N}(\beta_0 + \mathbf{\beta}^T \mathbf{x}_i, \sigma^2)$$

### 2. Prior Probability Bounds

To close the system mathematically under a standard regularized framework, we place weakly informative independent Normal priors on the intercept and slope coefficients, and a half-Normal distribution constraint on the scale parameter:

$$\beta_0 \sim \mathcal{N}(\mu_0, \sigma_0^2), \quad \beta_j \sim \mathcal{N}(\mu_j, \sigma_j^2), \quad \sigma \sim \text{Half-Normal}(\sigma_s)$$

---

## 10.2 Computational Implementation: Linear Parameter Fields

The script below builds a continuous linear regression model using `PyMC`. It estimates the joint posterior distribution via NUTS sampling and applies posterior predictive sampling to visualize the distribution of regression lines.

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# 1. Deterministic Synthetic Data Generation
np.random.seed(314)
sample_size = 20
x = np.linspace(0, 10, sample_size)
# Underlying true process: Intercept = 0.0, Slope = 2.5, Sigma = 2.0
true_sigma = 2.0
y = 2.5 * x + np.random.normal(loc=0.0, scale=true_sigma, size=sample_size)

# 2. Probabilistic Model Architectural Configuration
with pm.Model() as linear_regression_model:
    # Prior elicitation for hyper-parameters
    intercept = pm.Normal("intercept", mu=0.0, sigma=10.0)
    slope = pm.Normal("slope", mu=0.0, sigma=10.0)
    sigma = pm.HalfNormal("sigma", sigma=5.0)
    
    # Conditional expectation equation mapping the deterministic hyperplane
    mu = intercept + slope * x
    
    # Likelihood function definition
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
    
    # 3. Execution of MCMC Sampling (NUTS Transition Kernel)
    trace_linear = pm.sample(draws=2000, tune=1000, return_inferencedata=True, random_seed=42)
    
    # 4. Posterior Predictive Sampling for Structural Validation
    posterior_predictive = pm.sample_posterior_predictive(trace_linear)

# 5. Visual Summary of Post-Sampling Computations
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

# Plotting individual marginal parameter posteriors
az.plot_posterior(trace_linear, var_names=["intercept", "slope", "sigma"], ax=axes[0])
axes[0].set_title("Marginal Parameter Estimations")

# Visualizing regression line distribution over empirical data arrays
axes[1].scatter(x, y, color="#1f4e79", zorder=5, label="Observed Empirical Data")
posterior_intercepts = trace_linear.posterior["intercept"].values.flatten()
posterior_slopes = trace_linear.posterior["slope"].values.flatten()

# Display a subset of sampled lines to visualize geometric uncertainty
thinning_factor = 100
for i in range(0, len(posterior_intercepts), thinning_factor):
    axes[1].plot(x, posterior_intercepts[i] + posterior_slopes[i] * x, color="#d9534f", alpha=0.05)

axes[1].set_title("Posterior Quantile Regression Hyperplanes")
axes[1].set_xlabel("Explanatory Dimension (x)")
axes[1].set_ylabel("Response Domain (y)")
axes[1].legend()

plt.tight_layout()
plt.show()

```

---

## 10.3 Bayesian Logistic Regression for Discrete Vector Subspaces

When the dependent tracking index is binary ($y_i \in \{0,1\}$), the linear mapping framework fails because continuous probability spaces must be mapped to a bounded interval $[0,1]$. Logistic regression solves this by using a non-linear link function to map the linear predictor to the log-odds scale.

### The Structural Logit Architecture

| Architectural Layer | Mathematical Formulation | Transformation Domain |
| --- | --- | --- |
| **Linear Predictor Field** | $\eta_i = \beta_0 + \mathbf{\beta}^T \mathbf{x}_i$ | maps to the unconstrained real space: $\eta_i \in (-\infty, \infty)$ |
| **Logit Link Transformation** | $p_i = \sigma(\eta_i) = \frac{1}{1 + e^{-\eta_i}}$ | Maps the real line to bounded probability vectors: $p_i \in [0, 1]$ |
| **Bernoulli Likelihood** | $y_i \sim \text{Bernoulli}(p_i)$ | Evaluates probability mass for binary observation outcomes: $y_i \in \{0, 1\}$ |

The conditional log-odds ratio is isolated by inverting the activation transformation function:

$$\log\left(\frac{p_i}{1 - p_i}\right) = \beta_0 + \beta_1 x_{1i} + \dots + \beta_d x_{di}$$

---

## 10.4 Computational Implementation: Discrete Estimation Mechanics

The model script below simulates a binary classification problem. It defines a Bernoulli likelihood linked to a linear function, then samples from the parameter space to extract unbiased estimates of the log-odds coefficients.

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# 1. Generate Synthetic Binary Observations via a True Sigmoidal Link
np.random.seed(123)
sample_size_discrete = 100
x_logistic = np.random.randn(sample_size_discrete)

true_beta0 = 0.5
true_beta1 = 2.0
latent_probabilities = 1 / (1 + np.exp(-(true_beta0 + true_beta1 * x_logistic)))
y_logistic_data = np.random.binomial(n=1, p=latent_probabilities)

# 2. Probabilistic Specification of the Generalized Linear Framework
with pm.Model() as logistic_regression_model:
    # Prior definitions across the log-odds space
    beta0 = pm.Normal("beta0", mu=0.0, sigma=10.0)
    beta1 = pm.Normal("beta1", mu=0.0, sigma=10.0)
    
    # Linear mapping directly bounded inside the Bernoulli logit-parameter argument
    linear_predictor = beta0 + beta1 * x_logistic
    
    # Bernoulli likelihood optimization using logit-p parametrization
    y_obs_logistic = pm.Bernoulli("y_obs", logit_p=linear_predictor, observed=y_logistic_data)
    
    # 3. Execution of MCMC Convergence Pipeline
    trace_logistic = pm.sample(draws=2000, tune=1000, return_inferencedata=True, random_seed=42)

# 4. Post-Estimation Summary Diagnostics
print("--- Logistic Coefficient Joint Estimation Metrics ---")
logistic_summary_table = az.summary(trace_logistic, var_names=["beta0", "beta1"])
print(logistic_summary_table)

# 5. Extract Inferred Density Distributions
az.plot_posterior(trace_logistic, var_names=["beta0", "beta1"])
plt.suptitle("Posterior Parameter Densities on the Log-Odds Scale", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

```
