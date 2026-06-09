# Chapter 6: Bayesian Hypothesis Testing & Model Comparison

This chapter formalizes the probabilistic criteria used to evaluate competing statistical hypotheses and parametric model structures. In contrast to classical significance testing, the Bayesian framework treats model selection as an extension of probability theory, evaluating the relative plausibility of hypotheses by computing marginal likelihood ratios.

---

## 6.1 The Mathematical Foundations of the Bayes Factor

Let $M_1$ and $M_2$ represent two competing statistical models or hypotheses defined over a shared or distinct parameter space. Given an observed empirical data vector $D$, the **Bayes factor** $BF_{12}$ (often denoted as $K$) is defined mathematically as the ratio of the integrated marginal likelihoods (or evidence) of the data under each respective model:

$$BF_{12} = \frac{P(D \mid M_1)}{P(D \mid M_2)} = \frac{\int_{\Theta_1} P(D \mid \theta_1, M_1) P(\theta_1 \mid M_1) d\theta_1}{\int_{\Theta_2} P(D \mid \theta_2, M_2) P(\theta_2 \mid M_2) d\theta_2}$$

By applying Bayes' theorem to the models themselves, we establish the connection between the prior odds and the posterior odds of the models:

$$\frac{P(M_1 \mid D)}{P(M_2 \mid D)} = \frac{P(D \mid M_1)}{P(D \mid M_2)} \times \frac{P(M_1)}{P(M_2)} = BF_{12} \times \frac{P(M_1)}{P(M_2)}$$

Under a uniform prior over the model space ($P(M_1) = P(M_2) = 0.5$), the posterior odds simplify to equal the Bayes factor precisely.

### Epistemological Interpretations of $BF_{12}$

| Bayes Factor Range | Formal Evidentiary Interpretation |
| --- | --- |
| **$BF_{12} > 1$** | The observed empirical data $D$ provide positive evidence supporting model $M_1$ over model $M_2$. |
| **$BF_{12} < 1$** | The observed empirical data $D$ provide positive evidence supporting model $M_2$ over model $M_1$. |
| **$BF_{12} \approx 1$** | The data exhibit equal plausibility under both specifications; the empirical evidence fails to differentiate between $M_1$ and $M_2$. |

### Structural Advantages over Frequentist $p$-values

* **Symmetrical Evidence Quantification:** Unlike classical Null Hypothesis Significance Testing (NHST), which can only reject or fail to reject a null state, the Bayes factor provides a mathematically coherent metric to accumulate and quantify direct evidence *in favor* of a null hypothesis (e.g., $BF_{12} = 0.10$ establishes that $M_2$ is exactly ten times more plausible than $M_1$).
* **Automatic Parsimony Induction:** The integration of the parameter space explicitly penalizes models with excessive parameterization. Overfitted or overly complex models scale down the average likelihood density across wide prior bounds, naturally enforcing a mathematical version of Occam's razor.

---

## 6.2 Analytical Derivation via the Savage-Dickey Density Ratio

When evaluating a nested point null hypothesis against a continuous alternative, evaluating the full multidimensional integration in the marginal likelihood denominator can be computationally challenging. Under certain conditions, we can employ the **Savage-Dickey density ratio**.

Let $M_1$ represent a sharp point null hypothesis restricting a parameter to a fixed boundary ($\theta = \theta_0$), and let $M_2$ denote the unconstrained alternative model. If the prior distribution for the nuisance parameters under $M_1$ is equal to the prior under $M_2$ conditioned on $\theta = \theta_0$, the Bayes factor $BF_{12}$ simplifies to the ratio of the posterior density to the prior density evaluated precisely at the null point under the alternative model:

$$BF_{12} = \frac{P(\theta = \theta_0 \mid D, M_2)}{P(\theta = \theta_0 \mid M_2)}$$

This eliminates the need to calculate separate marginal likelihood integrations, reducing the evaluation to localized density point checks on the alternative distribution model.

---

## 6.3 Computational Implementation: Savage-Dickey Density Computation

The script below implements a formal Savage-Dickey calculation utilizing the beta-binomial conjugate models derived in prior chapters. We explicitly evaluate the point null hypothesis $M_1: p = 0.5$ (Fair Coin) against the unconstrained alternative $M_2: p \sim \text{Beta}(2,2)$ given empirical tracking records showing $n=10$ trials with $k=7$ successes (yielding an analytical posterior of $\text{Beta}(9,5)$).

```python
import numpy as np
from scipy.stats import beta

# 1. Structural Parameter Initialization
# Testing a localized sharp point null hypothesis against an unconstrained space
p_null = 0.5

# Model 2 Prior Parametrization: Beta(2, 2)
alpha_prior, beta_prior = 2, 2

# Model 2 Posterior Parametrization (Conditioned on 7 Successes, 3 Failures)
alpha_posterior = alpha_prior + 7
beta_posterior = beta_prior + 3

# 2. Localized Density Evaluations at the Null Coordinate Point
# Extracting exact mathematical density heights via the continuous PDF
posterior_density_at_null = beta.pdf(p_null, a=alpha_posterior, b=beta_posterior)
prior_density_at_null = beta.pdf(p_null, a=alpha_prior, b=beta_prior)

# 3. Savage-Dickey Ratio Construction
# BF_01 represents the Bayes Factor in favor of the Null (M1) over the Alternative (M2)
BF_01 = posterior_density_at_null / prior_density_at_null

# Inverse translation to obtain evidence in favor of the alternative framework
BF_10 = prior_density_at_null / posterior_density_at_null

# 4. Formally Format Statistical Estimators and Structural Frameworks
print("--- Savage-Dickey Rational Evidentiary Matrix ---")
print(f"Prior Target Point Coordinate Density H(p=0.5):      {prior_density_at_null:.4f}")
print(f"Posterior Target Point Coordinate Density H(p=0.5):  {posterior_density_at_null:.4f}")
print("-------------------------------------------------")
print(f"Bayes Factor in Favor of Null Model M_1 (BF_01):     {BF_01:.4f}")
print(f"Bayes Factor in Favor of Alternative M_2 (BF_10):    {BF_10:.4f}")
print("-------------------------------------------------")

# 5. Coherent Decoupled Decision Output Logic
if BF_01 > 1.0:
    print(f"Structural Inference: The data support the Sharp Point Null Model (M_1) by a factor of {BF_01:.2f}:1.")
else:
    print(f"Structural Inference: The data support the Unconstrained Alternative (M_2) by a factor of {BF_10:.2f}:1.")

```
