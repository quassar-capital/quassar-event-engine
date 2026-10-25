# Research 02 — High-Dimensional Event Study Inference

**Quassar Capital | Quantitative Research & Scientific Engineering**

## Abstract

This research program develops a reproducible framework for statistical inference on financial event responses in high-dimensional, cross-sectionally dependent panels. It addresses abnormal-return estimation, joint inference, overlapping event windows, covariance regularization, resampling-based uncertainty quantification, and multiple-hypothesis testing.

The central objective is to distinguish statistically defensible event-associated responses from ordinary variation without treating statistical association as proof of causation.

## 1. Research Objectives

The program investigates:

1. Estimation of abnormal and cumulative abnormal returns relative to explicitly specified expected-return models.
2. Cross-sectional and temporal dependence among financial instruments and event observations.
3. Numerically stable covariance estimation and generalized least-squares inference.
4. Finite-sample behavior of conventional and dependence-aware tests.
5. Event-window overlap, concurrent announcements, and contamination diagnostics.
6. Family-wise error rate and false discovery rate control across hypothesis families.
7. Reproducible Monte Carlo evaluation of empirical size, power, confidence-interval coverage, and computational performance.

## 2. Statistical Model

Let R(i,t) denote the return of instrument i at time t, and let F(t) be a vector of observable risk factors.

The expected-return model is:

    R(i,t) = alpha(i) + beta(i)' F(t) + epsilon(i,t)

Estimated abnormal returns are:

    AR(i,t) = R(i,t) - R_hat(i,t)

For an event window W, cumulative abnormal returns are:

    CAR(i,W) = sum_{t in W} AR(i,t)

Parameter estimation must distinguish pre-event estimation observations from event-window observations and explicitly account for estimation uncertainty where relevant.

## 3. Dependence and High-Dimensional Inference

Let Sigma denote the covariance matrix of the relevant error vector.

For an appropriately specified stacked linear model, feasible generalized least squares takes the form:

    theta_hat = (X' Sigma_hat^{-1} X)^{-1} X' Sigma_hat^{-1} y

Implementations must avoid explicit matrix inversion where numerically stable linear solves are available.

The research will evaluate covariance estimators under varying sample sizes, dimensions, correlation strengths, and conditioning regimes. Regularization may improve numerical stability but must not be interpreted as automatically establishing valid statistical inference.

## 4. Resampling and Multiple Testing

Candidate inferential procedures include dependence-aware bootstrap methods, cluster-based resampling, and stepdown multiple-testing corrections.

Every procedure must document:

- Its null hypothesis and target estimand.
- The assumed dependence structure.
- Required regularity or exchangeability conditions.
- The resampling unit and construction.
- Known limitations and failure regimes.

Multiple-testing evaluation will distinguish family-wise error rate from false discovery rate and assess whether nominal guarantees are supported under the simulated designs.

## 5. Identification and Event Contamination

Statistical significance alone does not establish causality.

The framework must explicitly identify overlapping event windows, concurrent announcements, potentially contaminated estimation periods, and specification sensitivity.

Causal language is permitted only when additional identification assumptions are stated and defended. Otherwise, reported results must be characterized as event-associated statistical evidence.

## 6. Scientific Architecture

The research will maintain independent implementations and validation responsibilities:

- Python: statistical models, experimental orchestration, and Monte Carlo evaluation.
- Rust: deterministic numerical and validation kernels.
- C++: performance-oriented computational kernels supported by benchmark evidence.
- Julia: independent mathematical reference implementations and numerical cross-checks.

Language diversity is not a substitute for scientific correctness. Cross-language agreement must be tested using explicit tolerances, reference cases, and reproducible fixtures.

## 7. Validation Protocol

The acceptance criteria include:

1. Analytic benchmark agreement where closed-form solutions exist.
2. Reproducibility under fixed random seeds.
3. Empirical type-I error measurement under specified null designs.
4. Power evaluation under controlled alternatives.
5. Confidence-interval coverage evaluation.
6. Robustness under cross-sectional correlation and overlapping events.
7. Numerical stability diagnostics for ill-conditioned covariance matrices.
8. Cross-language consistency checks.
9. Automated tests and continuous-integration execution.
10. Documented limitations, assumptions, and failure cases.

Monte Carlo findings must report simulation counts and uncertainty around estimated rejection or coverage rates. A failed nominal guarantee must be reported as a research result rather than suppressed.

## 8. Deliverables

The program will provide:

- Documented statistical estimators and inference procedures.
- Deterministic test fixtures and numerical reference cases.
- Monte Carlo experiments with reproducible configurations.
- Multiple-testing and dependence diagnostics.
- Independent implementation comparisons.
- Benchmark results with stated hardware and software conditions.
- A scientific report covering methodology, evidence, limitations, and interpretation.

## 9. Research Boundaries

This program does not implement portfolio optimization, trading execution, market-impact modeling, factor discovery, scenario forecasting, data-integrity infrastructure, or online change-point detection.

Its scope is statistical inference on financial event responses.

## 10. Scientific Integrity

The project must not claim causal identification, universal finite-sample validity, superior performance, or original mathematical discovery without supporting evidence.

All statistical conclusions must be traceable to explicit assumptions, reproducible calculations, and independently testable results.

---

**Research Program:** 02  
**Repository:** quassar-event-engine  
**Institution:** Quassar Capital  
**Discipline:** Financial Econometrics, High-Dimensional Statistics, Scientific Computing
