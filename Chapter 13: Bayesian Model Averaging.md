# Chapter 13: Bayesian Model Averaging

In preceding chapters, model selection was approached by identifying a singular optimal parametric structure using localized hypothesis testing or point boundaries. However, restricting inference to a solitary candidate specification introduces systematic model risk and ignores structural uncertainty. This chapter establishes the mathematical framework of **Bayesian Model Averaging (BMA)**, a paradigm where out-of-sample predictive tasks integrate across an entire ensemble space of plausible models, weighted by their posterior probabilities.

---

## 13.1 Epistemological Limitations of Singular Model Selection

When an investigator selects a single "best" model based on an information criterion or a $p$-value threshold, subsequent inferences are conditioned on the assumption that this selected model is the exact data-generating mechanism. This approach suffers from two distinct flaws:

1. **Overconfidence Bias:** Standard errors and credible intervals calculated from a single pre-selected model are systematically too narrow because they account for parameter uncertainty but ignore structural model uncertainty.
2. **Arbitrary Thresholding:** If two competing candidate models exhibit highly similar explanatory power, selecting one over the other based on minor statistical variations can invalidate downstream empirical predictions.

---

## 13.2 Mathematical Formulation of Model Averaging

Let $\mathcal{M} = \{M_1, M_2, \dots, M_K\}$ define the discrete space of all candidate models under consideration. Let $\Delta$ denote the target quantity of interest (such as a future out-of-sample observation or a latent structural parameter present across specifications).

The joint posterior probability distribution of $\Delta$ given the observed empirical data matrix $D$ is obtained by applying the law of total probability across the model space:

$$P(\Delta \mid D) = \sum_{k=1}^{K} P(\Delta \mid M_k, D) P(M_k \mid D)$$

Where the weight assigned to each individual model's prediction is its exact **posterior model probability** $P(M_k \mid D)$, derived via Bayes' theorem:

$$P(M_k \mid D) = \frac{P(D \mid M_k) P(M_k)}{\sum_{j=1}^{K} P(D \mid M_j) P(M_j)}$$

The term $P(D \mid M_k)$ represents the marginal likelihood (or integrated evidence) of model $M_k$, which integrates out the specific parameter vector $\boldsymbol{\theta}_k$:

$$P(D \mid M_k) = \int_{\Theta_k} P(D \mid \boldsymbol{\theta}_k, M_k) P(\boldsymbol{\theta}_k \mid M_k) d\boldsymbol{\theta}_k$$

---

## 13.3 Predictive Accuracy Metrics: WAIC and LOO

Because calculating the exact multidimensional marginal likelihood $P(D \mid M_k)$ is computationally intractable for complex non-conjugate spaces, modern Bayesian frameworks utilize information theoretic criteria to estimate out-of-sample predictive accuracy directly from MCMC samples.

### 1. Watanabe-Akaike Information Criterion (WAIC)

WAIC is a fully Bayesian criterion that evaluates predictive accuracy pointwise, bypassing asymptotic assumptions. It uses the log pointwise predictive density ($\text{lppd}$) and adjusts for model complexity using a penalty term $p_{\text{WAIC}}$ calculated from the variance of the log likelihoods across the MCMC chain:

$$\text{WAIC} = -2 \left( \text{lppd} - p_{\text{WAIC}} \right)$$

### 2. Leave-One-Out Cross-Validation (LOO-CV)

LOO estimates out-of-sample predictive capacity by sequentially removing a single observation $y_i$ and evaluating the model conditioned on the remaining $n-1$ data points. To avoid fitting the model $n$ separate times, modern implementations use **Pareto-Smoothed Importance Sampling (PSIS-LOO)** to approximate the exact cross-validation profile using standard posterior draws:

$$\text{LOO} = -2 \sum_{i=1}^n \log \int_{\Theta} P(y_i \mid \boldsymbol{\theta}) P(\boldsymbol{\theta} \mid y_{-i}) d\boldsymbol{\theta}$$

### Structural Metric Comparison Matrix

| Informational Index | Primary Optimization Focus | Mathematical Constraints | Interpretation Rule |
| --- | --- | --- | --- |
| **WAIC** | Pointwise log-predictive density over the sample space. | Can become unstable if posterior distributions have highly unequal weight profiles or outliers. | Minimization objective; lower absolute scores indicate superior predictive performance. |
| **PSIS-LOO** | Out-of-sample prediction via importance sampling approximations. | Monitored via the shape parameter ($\kappa$) of the Pareto distribution; values of $\kappa > 0.7$ indicate unreliable approximations. | Minimization objective; provides a robust assessment of predictive validation. |

---

## 13.4 Computational Implementation: Predictive Comparisons

The program below implements a formal predictive comparison between the multi-parameter continuous linear regression model developed in Section 10.2 ($M_1$: Linear Predictor) and a nested, unpooled alternative specification ($M_2$: Intercept-Only Framework). We use `PyMC` to define the baseline structures and evaluate their relative predictive strength using `ArViZ`.

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Operational Assumptions:
# 'x' and 'y' represent the continuous data arrays initialized in Section 10.2.
# 'trace_linear' represents the converged InferenceData object from the linear model.

# 1. Structural Specification of the Intercept-Only Counter-Model (M_2)
with pm.Model() as intercept_only_model:
    # Baseline location and scale priors
    intercept = pm.Normal("intercept", mu=np.mean(y), sigma=5.0)
    sigma = pm.HalfNormal("sigma", sigma=5.0)
    
    # Likelihood formulation lacking covariate mapping dimensions
    y_obs = pm.Normal("y_obs", mu=intercept, sigma=sigma, observed=y)
    
    # Execute MCMC convergence routine
    trace_intercept_only = pm.sample(
        draws=1000, 
        tune=1000, 
        return_inferencedata=True, 
        random_seed=42
    )

# 2. Compute Predictive Metric Matrices via Information Criteria
with linear_regression_model:
    # Calculate LOO-CV values for the alternative models
    # Note: Ensure linear_regression_model is the active context block
    pass 

# 3. Compile Comparative Frame Layouts
model_dictionary = {
    "Covariate Linear Model (M_1)": trace_linear,
    "Intercept-Only Base Model (M_2)": trace_intercept_only
}

comparison_dataframe = az.compare(model_dictionary, ic="loo")
print("--- Information Theoretic Comparative Matrix ---")
print(comparison_dataframe)

# 4. Generate Graphical Out-of-Sample Diagnostics
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_compare(comparison_dataframe, ax=ax)
ax.set_title(
    "Information Criteria Space: Out-of-Sample Deviance Mappings", 
    fontsize=12, 
    fontweight="bold"
)
plt.tight_layout()
plt.show()

```

### Analysis of the Comparative Metrics

The tabular results generated by `az.compare` rank candidate specifications based on estimated predictive deviance. The model position with the lowest relative value (typically assigned a rank index of 0) represents the preferred model structure.

The `weight` column in the summary table approximates the posterior model probabilities $P(M_k \mid D)$ using Pseudo-BMA weights. If the covariate model ($M_1$) receives a weight profile approaching $1.0$, it indicates that including the predictor variable provides a substantial improvement in out-of-sample predictive accuracy compared to the intercept-only baseline model ($M_2$).
