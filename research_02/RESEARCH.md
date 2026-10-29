
# Research 02 — High-Dimensional Event Study Inference

**Quassar Capital | Quantitative Research & Scientific Engineering**

**Research Discipline:** Financial Econometrics, High-Dimensional Statistical Inference, Dependence Modeling, and Scientific Computing

## Abstract

This research program examines statistical inference for financial event responses in multivariate environments characterized by cross-sectional dependence, serial correlation, heterogeneous signal strength, and simultaneous hypothesis testing.

The computational framework integrates abnormal-return analysis, dependence-aware uncertainty estimation, heteroskedasticity and autocorrelation consistent (HAC) studentization, block-bootstrap resampling, and family-wise multiple-testing procedures.

Its central contribution is a structured scientific implementation for examining event-associated financial responses while explicitly separating statistical evidence, numerical verification, and economic interpretation.

The research combines Python-based statistical computation with independent numerical implementations in Rust, C++17, and Julia. Reproducible simulation designs provide a controlled environment for examining inferential behavior under specified dependence structures.

## 1. Scientific Objectives

The research addresses seven interconnected problems:

1. Estimating event-associated abnormal returns relative to explicitly defined expected-return models.
2. Characterizing cross-sectional and temporal dependence in multivariate financial observations.
3. Constructing numerically stable statistical procedures for correlated observations.
4. Evaluating studentized statistics under heteroskedasticity and serial dependence.
5. Implementing resampling-based joint inference for multiple hypotheses.
6. Measuring the empirical behavior of simultaneous testing procedures under complete-null and partial-null configurations.
7. Establishing reproducible computational procedures across independently implemented numerical environments.

These objectives are treated as distinct statistical and computational questions. Numerical agreement, simulation performance, and theoretical inferential validity are evaluated according to their respective evidentiary requirements.

## 2. Mathematical Framework

### 2.1 Expected Returns and Event-Associated Responses

Let R(i,t) represent the observed return of instrument i at time t, and let F(t) denote a vector of observable risk factors.

A linear expected-return specification is

    R(i,t) = alpha(i) + beta(i)' F(t) + epsilon(i,t)

The estimated abnormal return is

    AR(i,t) = R(i,t) - R_hat(i,t)

For an event window W, cumulative abnormal returns are defined as

    CAR(i,W) = sum_{t in W} AR(i,t)

The statistical interpretation of these quantities depends on the expected-return specification, estimation sample, event-window definition, and information available at the event timestamp.

An abnormal return is a model-relative deviation. It does not independently establish that the associated event caused the observed market movement.

### 2.2 Joint Statistical Representation

Consider a panel of n observations across p instruments, with the corresponding vector of event-associated responses denoted by Y(t).

The general inferential problem concerns a family of hypotheses

    H0(i): mu(i) = 0,  i = 1,...,p

where mu(i) denotes the population mean of the specified response for instrument i.

For a studentized mean statistic,

    T(i) = sqrt(n) * mean(Y(i)) / s_hat(i)

the estimated standard error s_hat(i) must account for the dependence structure assumed by the selected inferential procedure.

Simultaneous inference is based on the joint behavior of the statistic vector

    T = [T(1), ..., T(p)]'

rather than on an isolated sequence of unadjusted marginal tests.

## 3. Dependence Modeling and Covariance Structure

### 3.1 Cross-Sectional Dependence

Financial instruments frequently respond to shared information, common risk factors, and correlated liquidity conditions.

Consequently, cross-sectional covariance cannot generally be ignored when interpreting a family of event-related hypotheses.

Let Sigma denote the covariance matrix of the relevant multivariate disturbance vector.

The statistical framework distinguishes covariance estimation from inferential validity: obtaining a numerically well-conditioned covariance estimate does not, by itself, establish the correctness of a hypothesis-testing procedure.

### 3.2 Serial Dependence and HAC Estimation

Temporal dependence changes the sampling variability of estimated means and related statistics.

For a stationary scalar process with autocovariance gamma(k), the long-run variance is

    Omega = gamma(0) + 2 * sum_{k=1}^{infinity} gamma(k)

A Bartlett-kernel HAC estimator takes the form

    Omega_hat = gamma_hat(0)
              + 2 * sum_{k=1}^{L}
                  (1 - k / (L + 1)) * gamma_hat(k)

where L is the selected truncation lag.

The corresponding standard error for a sample mean is obtained from the estimated long-run variance and sample size.

HAC estimation requires attention to bandwidth selection, sample length, dependence persistence, and the properties of the underlying process.

### 3.3 Numerical Linear Algebra

For a properly specified stacked linear model, feasible generalized least squares can be expressed as

    theta_hat =
        (X' Sigma_hat^(-1) X)^(-1)
        X' Sigma_hat^(-1) y

This expression defines the estimator mathematically; a reliable numerical implementation should prefer stable matrix factorizations and linear solves to explicit inversion.

Covariance conditioning, near-singularity, finite precision, and regularization are separate considerations from the validity of subsequent statistical inference.

## 4. Simultaneous Inference and Resampling

### 4.1 Maximum-Statistic Inference

For p studentized statistics, define the maximum absolute statistic

    M = max_{1 <= i <= p} |T(i)|

A resampling procedure constructs an approximation to the distribution of M under an appropriately specified null mechanism.

If c_alpha denotes the estimated upper critical value, a single-step simultaneous decision rule rejects H0(i) when

    |T(i)| > c_alpha

This construction incorporates the joint behavior of the tested statistics rather than applying independent marginal thresholds.

The interpretation of the resulting rejection decisions depends on the validity of the resampling approximation under the data-generating process.

### 4.2 Dependence-Aware Block Resampling

The computational research uses block-based resampling to examine joint statistics when observations exhibit temporal dependence.

Circular block resampling draws consecutive observation blocks with wraparound indexing, preserving local temporal structure within each sampled block.

The selected block length influences the approximation. Short blocks may inadequately represent persistent dependence, while long blocks reduce the number of effectively distinct resampled segments.

Block-bootstrap validity therefore depends on the dependence regime, resampling construction, sample size, block-length selection, and studentization procedure.

### 4.3 Multiple-Testing Interpretation

For a family of hypotheses, the family-wise error rate is

    FWER = P(V >= 1)

where V is the number of rejected true null hypotheses.

The distinction between complete-null and partial-null configurations is essential.

Under a complete null, all tested hypotheses are true. Under a partial null, true null hypotheses coexist with nonzero alternatives.

The two settings answer different scientific questions and must not be treated as interchangeable evidence of simultaneous error control.

## 5. Computational Research Architecture

The implementation is organized around separate statistical and numerical responsibilities.

### Python — Statistical Computation

Python provides the research environment for statistical models, simulation orchestration, dependence configurations, and reproducible evaluation.

The Research 02 implementation includes dedicated modules for complete-null and partial-null experiments, together with a deterministic calibration entry point.

### Rust — Numerical Verification

Rust provides an independent implementation environment for deterministic numerical operations and verification cases.

Its role is to support explicit computational invariants, controlled numerical behavior, and independently inspectable reference calculations.

### C++17 — Scientific Numerical Computing

C++17 provides a separate numerical implementation suitable for examining algorithmic behavior in a compiled computing environment.

Performance claims require direct benchmark evidence and are not inferred solely from the programming language.

### Julia — Mathematical Cross-Checks

Julia provides an additional scientific computing implementation for numerical comparison and mathematical reference calculations.

Independent language implementations support verification of selected computational results. They do not automatically establish full equivalence between the statistical capabilities of the different implementations.

## 6. Reproducible Calibration Design

The research includes a deterministic Monte Carlo calibration framework for examining the behavior of a studentized Max-T procedure under specified Gaussian panel models.

The calibration configuration uses:

| Parameter | Specification |
|---|---|
| Observations per simulated panel | 120 |
| Financial instruments | 8 |
| Monte Carlo replications | 300 |
| Bootstrap resamples | 299 |
| Circular block length | 6 |
| Nominal significance level | 5% |
| Random seed | 2026 |
| Temporal model | Gaussian AR(1) |
| Cross-sectional structure | Equicorrelated innovations |
| Studentization | Bartlett HAC |
| Joint testing procedure | Single-step Max-T |

### 6.1 Complete-Null Configurations

Three dependence regimes are specified:

| Configuration | Temporal correlation | Cross-sectional correlation |
|---|---:|---:|
| Independent | 0.0 | 0.0 |
| Moderate dependence | 0.3 | 0.4 |
| Strong dependence | 0.7 | 0.8 |

These configurations examine changes in empirical rejection behavior as temporal persistence and cross-sectional correlation increase.

### 6.2 Partial-Null Configuration

The partial-null design introduces three nonzero alternatives among eight instruments.

Its specified parameters include:

- Alternative effect size: 0.5
- Temporal correlation: 0.3
- Cross-sectional correlation: 0.4
- The same observation count, resampling count, block length, significance level, and random seed as the complete-null configurations.

The partial-null setting permits examination of false-family rejections when genuine nonzero responses coexist with true null hypotheses.

### 6.3 Statistical Precision

For an empirical rejection proportion p_hat computed from B independent Monte Carlo replications, the conventional binomial Monte Carlo standard-error estimate is

    MCSE = sqrt(p_hat * (1 - p_hat) / B)

This quantity describes simulation uncertainty in the estimated rejection frequency.

It does not quantify all sources of inferential error, establish bootstrap consistency, or provide a universal guarantee of nominal error control.

## 7. Verification and Scientific Evidence

The project distinguishes three levels of evidence.

### 7.1 Numerical Verification

Deterministic numerical cases evaluate selected computational operations against independently specified expectations.

These checks address implementation consistency and numerical correctness within their tested scope.

### 7.2 Automated Software Validation

Automated workflows verify designated software components and their associated tests.

The Research 02 multilingual validation workflow exercises the independent Rust, C++17, and Julia components.

Successful workflow execution establishes that the specified checks passed in the configured environment. It is not a substitute for a mathematical proof of statistical validity.

### 7.3 Statistical Calibration

Monte Carlo evaluation examines empirical rejection behavior under explicitly defined stochastic models.

The interpretation is design-specific: simulation evidence under stationary Gaussian AR(1) panels with equicorrelated innovations cannot automatically be extended to arbitrary financial data-generating processes.

Calibration findings are evaluated in relation to the nominal significance level, simulation uncertainty, dependence configuration, and selected resampling parameters.

The framework is intended for rigorous examination of statistical procedures, including identification of configurations requiring further methodological refinement.

## 8. Event Identification and Interpretation

The research distinguishes event-associated statistical relationships from causal effects.

A defensible event study requires explicit treatment of:

- Event occurrence and public-information timestamps.
- Estimation windows and event windows.
- Expected-return model specification.
- Concurrent announcements and overlapping event windows.
- Cross-sectional correlation and serial dependence.
- Multiple-hypothesis testing.
- Selection effects and potential information leakage.

Statistical significance is not sufficient evidence of causal attribution.

Any causal interpretation requires additional identifying assumptions appropriate to the event and the observed market environment.

## 9. Research Contributions

The scientific and engineering contributions of Research 02 comprise:

1. A mathematical framework for multivariate event-associated statistical inference.
2. Explicit treatment of temporal and cross-sectional dependence in simultaneous testing.
3. Studentized maximum-statistic inference with block-based resampling.
4. Separate complete-null and partial-null simulation configurations.
5. Deterministic calibration specifications supporting reproducible evaluation.
6. Independent numerical implementation components in Rust, C++17, and Julia.
7. Automated validation infrastructure for designated computational components.
8. A disciplined distinction between numerical verification, empirical calibration, and statistical validity.

The program emphasizes the quality of its mathematical formulation, computational organization, and evidentiary discipline rather than unsupported claims of universal performance.

## 10. Scientific Scope and Research Integrity

Research 02 is specifically concerned with high-dimensional event-study inference and dependence-aware statistical evaluation.

It is distinct from online change-point detection, portfolio construction, trading execution, factor discovery, scenario forecasting, and market microstructure modeling.

The scope of any empirical conclusion is determined by the assumptions and configurations under which its supporting evidence was obtained.

In particular:

- Simulation-based calibration is not equivalent to a universal finite-sample theorem.
- Cross-language numerical agreement is not equivalent to statistical identification.
- Correlation-adjusted inference does not independently establish causality.
- Numerical stability does not automatically establish statistical validity.
- Research software validation does not imply production trading-system certification.

These distinctions form part of the methodological foundation of the program and support precise interpretation of its scientific contributions.

---

**Research Program:** 02  
**Research Title:** High-Dimensional Event Study Inference  
**Repository:** quassar-event-engine  
**Institution:** Quassar Capital  
**Scientific Domains:** Financial Econometrics, High-Dimensional Statistics, Statistical Computing, Numerical Analysis
