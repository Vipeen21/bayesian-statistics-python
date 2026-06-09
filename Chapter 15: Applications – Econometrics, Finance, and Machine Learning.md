# Chapter 15: Applications – Econometrics, Finance, and Machine Learning

This concluding chapter demonstrates the practical application of Bayesian inference across three quantitative disciplines: Econometrics, Quantitative Finance, and Machine Learning. By transitioning from theoretical foundations to empirical execution, we examine how the Bayesian framework addresses structural complexities, handles parameter uncertainty, and mitigates overfitting in high-dimensional systems.

---

## 15.1 Econometrics

Econometric modeling frequently requires analyzing time-series and panel data structures where multi-parameter architectures are susceptible to severe overfitting under classical maximum likelihood estimation. Bayesian methods introduce formal prior structures to regularize these parameter spaces.

### 1. Bayesian Vector Autoregression (BVAR)

Vector Autoregressive (VAR) models capture the joint dynamic interactions across multiple endogenous time series. However, an unconstrained $\text{VAR}(p)$ model with $M$ variables scales its parameter count at a rate of $\mathcal{O}(M^2 p)$, leading to rapid degrees-of-freedom exhaustion. A Bayesian VAR overcomes this bottleneck by applying regularizing shrinkage priors—such as modifications of the classical Minnesota prior—which shrink the parameter coefficients toward a parsimonious random walk specification.

Let $\mathbf{Y}_t$ be an $M \times 1$ vector of endogenous variables. A $\text{VAR}(1)$ process is formulated as:

$$\mathbf{Y}_t = \mathbf{\Phi}_1 \mathbf{Y}_{t-1} + \mathbf{\epsilon}_t, \quad \mathbf{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma})$$

```python
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# 1. Stochastic Simulation of a Bivariate VAR(1) Process
np.random.seed(42)
T = 100
A1_true = np.array([[0.7, 0.2], [-0.1, 0.9]])
cov_true = np.array([[1.0, 0.3], [0.3, 1.5]])

Y = np.zeros((T, 2))
for t in range(1, T):
    Y[t, :] = A1_true @ Y[t-1, :] + np.random.multivariate_normal([0, 0], cov_true)

# 2. Probabilistic Model Configuration
with pm.Model() as bvar_model:
    # Prior mapping on the transition matrix enforcing zero-directed shrinkage
    A1 = pm.Normal("A1", mu=0, sigma=0.5, shape=(2, 2))
    
    # Covariance factorization utilizing an LKJ Cholesky decomposition kernel
    chol, _, _ = pm.LKJCholeskyCov(
        "chol", 
        n=2, 
        eta=2.0, 
        sd_dist=pm.Exponential.dist(1.0, shape=2)
    )
    cov = pm.Deterministic("cov", chol @ chol.T)
    
    # Conditional Mean Matrix Multiplication
    mu = pm.math.dot(A1, Y[:-1, :].T).T
    
    # Multivariate Normal Likelihood Specification
    likelihood = pm.MvNormal("likelihood", mu=mu, chol=chol, observed=Y[1:, :])
    trace_bvar = pm.sample(1000, tune=1500, random_seed=42)

# 3. Post-Estimation Output Display
print("True Transition Parameter Matrix A1:\n", A1_true)
print("\nPosterior Mean Estimation Matrix A1:\n", trace_bvar.posterior['A1'].mean(axis=(0, 1)).values)

```

### 2. Hierarchical Model for Panel Data

Panel (longitudinal) structures track multiple cross-sectional units $j \in \{1, \dots, J\}$ across synchronous time dimensions $t \in \{1, \dots, T\}$. Hierarchical Bayesian parameterization handles group heterogeneity through partial pooling, enabling data-sparse individual units to borrow statistical strength from the global population distribution.

The data-generating process with group-specific intercepts is defined as:

$$y_{it} = \alpha_{j[i]} + \beta x_{it} + \epsilon_{it}, \quad \epsilon_{it} \sim \mathcal{N}(0, \sigma_y^2)$$

$$\alpha_j \sim \mathcal{N}(\mu_a, \sigma_a^2)$$

```python
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# 1. Panel Data Matrix Simulation (5 Groups across 20 Epochs)
np.random.seed(123)
n_stores = 5
n_years = 20
store_idx = np.repeat(np.arange(n_stores), n_years)

store_intercepts_true = np.random.normal(50, 10, n_stores)
slope_true = 2.5
X = np.tile(np.arange(n_years), n_stores)
y = store_intercepts_true[store_idx] + slope_true * X + np.random.normal(0, 5, size=n_stores*n_years)

# 2. Multilevel Probabilistic Architecture
with pm.Model() as panel_model:
    # Hyperprior parameter boundaries for population distribution
    mu_a = pm.Normal('mu_a', mu=50, sigma=10)
    sigma_a = pm.HalfNormal('sigma_a', sigma=10)
    
    # Common fixed-effect slope parameter
    b = pm.Normal('b', mu=0, sigma=5)
    
    # Group-specific random intercepts
    a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_stores)
    sigma_y = pm.HalfNormal('sigma_y', sigma=10)
    
    mu = a[store_idx] + b * X
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma_y, observed=y)
    trace_panel = pm.sample(1000, tune=1000, random_seed=42)

```

### 3. State-Space Models (Dynamic Linear Models)

State-space models track the evolution of unobserved, latent state vectors over time. The Bayesian approach treats both the latent path and the system variances as joint random variables, estimating them simultaneously through recursive conditioning.

The local level model configuration is formulated as:

$$\text{Observation Equation: } y_t = \mu_t + v_t, \quad v_t \sim \mathcal{N}(0, \sigma_{\text{obs}}^2)$$

$$\text{System Equation: } \mu_t = \mu_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, \sigma_{\text{level}}^2)$$

```python
# 1. Structural Model Definition using Gaussian Random Walks
with pm.Model() as dlm_model:
    # Variance hyperpriors
    sigma_level = pm.HalfNormal("sigma_level", sigma=50)
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=100)
    
    # Latent state trajectory tracking via a random walk prior
    # Assuming 'nile_flow' is a pre-loaded 1D data array
    level = pm.GaussianRandomWalk(
        "level", 
        mu=0, 
        sigma=sigma_level, 
        shape=len(nile_flow)
    )
    
    likelihood = pm.Normal("likelihood", mu=level, sigma=sigma_obs, observed=nile_flow)
    trace_dlm = pm.sample(2000, tune=2000, random_seed=42)

```

---

## 15.2 Finance

Financial risk optimization often breaks down when relying on singular point estimates, as standard estimators tend to mistake sampling noise for structural economic signals. The Bayesian framework incorporates parameter uncertainty directly into the asset allocation and portfolio optimization loop.

### 1. Bayesian Portfolio Optimization

The classic Markowitz mean-variance optimization engine is highly sensitive to input errors. Small variations in sample means ($\hat{\mathbf{\mu}}$) can cause drastic shifts in optimal portfolio weights, leading to unstable allocations. By optimizing over the entire joint posterior predictive distribution instead of singular point configurations, the Bayesian approach naturally regularizes asset allocations to match the true state of parameter uncertainty.

The objective maximizes the expected return adjusted for variance given the posterior asset distributions:

$$\max_{\mathbf{w}} \quad \mathbf{w}^T \mathbb{E}[\mathbf{\mu} \mid D] - \frac{\gamma}{2} \mathbf{w}^T \mathbb{E}[\mathbf{\Sigma} \mid D] \mathbf{w}, \quad \text{subject to } \mathbf{w}^T \mathbf{1} = 1$$

```python
import numpy as np
import pymc as pm
import arviz as az

# 1. Return Matrix Simulation
np.random.seed(42)
true_means = np.array([0.05, 0.08])
true_cov = np.array([[0.01, 0.0075], [0.0075, 0.0225]])
returns = np.random.multivariate_normal(true_means, true_cov, size=20)

# 2. Bayesian Estimation of Location and Scale
with pm.Model() as portfolio_model:
    packed_chol = pm.LKJCholeskyCov("packed_chol", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0))
    chol = pm.expand_packed_triangular(2, packed_chol)
    mu = pm.Normal("mu", mu=0, sigma=0.2, shape=2)
    
    obs = pm.MvNormal("obs", mu=mu, chol=chol, observed=returns)
    trace_portfolio = pm.sample(1000, tune=1000, random_seed=42)

# 3. Allocation Integration Over the Extracted Posterior Space
posterior_samples = az.extract(trace_portfolio)
optimal_weights = []

for m, c in zip(posterior_samples['mu'].values.T, posterior_samples['cov'].values.T):
    inv_cov = np.linalg.inv(c)
    w_unnormalized = inv_cov @ m
    optimal_weights.append(w_unnormalized / np.sum(w_unnormalized))

```

### 2. Bayesian Risk Management (Value-at-Risk)

Value-at-Risk ($\text{VaR}_\alpha$) defines the empirical quantile threshold where the probability of a portfolio's losses exceeding that bound is strictly restricted to $\alpha$. A Bayesian approach evaluates VaR by drawing from the comprehensive posterior predictive distribution, ensuring that downstream capital reserves explicitly account for parameter risk and heavy-tailed distribution profiles.

$$\int_{-\infty}^{\text{VaR}_\alpha} P(y^* \mid D) dy^* = \alpha$$

```python
# 1. Student-t Likelihood Modeling to Capture Fat-Tailed Returns
with pm.Model() as var_model:
    mu = pm.Normal('mu', mu=0, sigma=0.01)
    sigma = pm.HalfNormal('sigma', sigma=0.05)
    # Degrees of freedom prior to capture tail kurtosis dynamically
    nu = pm.Gamma('nu', alpha=2, beta=0.1)
    
    likelihood = pm.StudentT('likelihood', nu=nu, mu=mu, sigma=sigma, observed=asset_returns)
    trace_var = pm.sample(1000, tune=1000, random_seed=42)
    ppc_var = pm.sample_posterior_predictive(trace_var, random_seed=42)

# 2. Compute Quantile Bounds over the Predictive Vector Space
future_returns = ppc_var.posterior_predictive['likelihood'].values.flatten()
VaR_95 = np.quantile(future_returns, 0.05)
print(f"95% Value-at-Risk Bound: {VaR_95 * 100:.3f}%")

```

---

## 15.3 Machine Learning

In modern machine learning applications, the Bayesian paradigm provides a framework for structured regularization, exact predictive uncertainty quantification, and flexible non-parametric functional estimation.

### 1. The Bayesian Lasso

Classical L1 regularization (Lasso) performs variable selection by adding an absolute penalty term to the cost function, forcing irrelevant coefficients to zero. In the Bayesian context, this behavior is replicated by placing independent Laplace (double-exponential) priors over the feature weight vector $\boldsymbol{\beta}$. This regularizes the parameter field and generates complete posterior credible intervals for every input feature.

$$\beta_j \sim \text{Laplace}(0, b) \propto \exp\left(-\frac{\lvert\beta_j\rvert}{b}\right)$$

```python
# 1. Sparse High-Dimensional Parameter Space Initialization
with pm.Model() as lasso_model:
    # Laplace prior induces sparsity over the K-dimensional feature space
    beta = pm.Laplace('beta', mu=0, b=1, shape=K)
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=5)
    
    mu = intercept + pm.math.dot(X, beta)
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y_lasso)
    trace_lasso = pm.sample(1000, tune=1500, random_seed=42)

```

### 2. Bayesian Neural Networks (BNN)

A Bayesian Neural Network treats all weight matrices and bias vectors as latent random variables rather than static point configurations. Instead of optimization routines finding a single local minimum (as in standard backpropagation), variational inference or sampling methods map the full joint posterior distribution over the network's parameters. This enables BNNs to distinguish between aleatoric uncertainty (inherent data noise) and epistemic uncertainty (model ignorance), allowing the network to signal when it encounters out-of-distribution inputs.

Let $f(\mathbf{x}; \mathbf{W})$ define the forward network map. The conditional class distribution is expressed as:

$$P(y_i = 1 \mid \mathbf{x}_i, \mathbf{W}) = \text{sigm}\left(f(\mathbf{x}_i; \mathbf{W})\right), \quad \mathbf{W} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

```python
# 1. Variational Approximation of Network Target Planes
with bnn_model:
    # Fitting the network parameter space via fast ADVI optimization
    variational_approx = pm.fit(n=30000, method='advi', random_seed=42)
trace_bnn = variational_approx.sample(1000)

```

### 3. Gaussian Processes (GPs)

Gaussian Processes represent a powerful non-parametric Bayesian framework for regression and function estimation. Instead of assuming a rigid parametric functional form (e.g., linear or polynomial), a GP defines a prior distribution directly over the space of continuous functions. This prior is fully characterized by a mean function $m(\mathbf{x})$ and a covariance kernel function $k(\mathbf{x}, \mathbf{x}')$, providing smooth interpolation and exact uncertainty quantification across unobserved design coordinates.

$$f(\mathbf{x}) \sim \mathcal{GP}\left(m(\mathbf{x}), \, k(\mathbf{x}, \mathbf{x}')\right)$$

```python
# 1. Functional Non-Parametric Estimation Loop
with pm.Model() as gp_model:
    length_scale = pm.Gamma("length_scale", alpha=2, beta=1)
    eta = pm.HalfCauchy("eta", beta=5)
    
    # Exponentiated Quadratic Covariance Kernel Specification
    cov_func = eta**2 * pm.gp.cov.ExpQuad(1, ls=length_scale)
    gp = pm.gp.Marginal(cov_func=cov_func)
    
    sigma = pm.HalfNormal("sigma", sigma=5)
    y_ = gp.marginal_likelihood("y", X=X_gp, y=y_gp, noise=sigma)
    trace_gp = pm.sample(1000, tune=1000, random_seed=42)

```

---

## 15.4 Summary Taxonomy of Bayesian Applied Frameworks

The following systematic index maps out the structural matching profiles of the core methodology components presented throughout this course material portfolio.

| Functional Application Domain | Latent Target Objective | Optimal Likelihood Selection | Structural Prior Strategy |
| --- | --- | --- | --- |
| **Dynamic Econometrics** | Joint Time-Series Interdependencies | Multivariate Normal ($\mathcal{M}v\mathcal{N}$) | Minnesota / High-Shrinkage Normal |
| **Panel Analysis** | Cross-Sectional Group Heterogeneity | Gaussian Component ($\mathcal{N}$) | Hierarchical Gaussian Population Hyperpriors |
| **Asset Allocation** | Portfolio Expected Return and Volatility | Multivariate Normal ($\mathcal{M}v\mathcal{N}$) | LKJ Cholesky Factorization Matrix |
| **Risk Containment** | Extremal Loss Quantile Boundary | Student-$t$ Distribution ($\text{StudentT}$) | Gamma Prior over Tail Degrees-of-Freedom ($\nu$) |
| **Feature Selection** | High-Dimensional Parameter Sparsity | Gaussian Component ($\mathcal{N}$) | Double-Exponential Laplace Kernel |
| **Deep Learning** | Structural Predictive Uncertainty | Bernoulli / Categorical Vector | Spherical Standard Gaussian ($\mathcal{N}(\mathbf{0}, \mathbf{I})$) |
| **Functional Estimation** | Non-Linear Non-Parametric Mapping | Gaussian Process Marginal | Covariance Kernels ($ExpQuad$) |
