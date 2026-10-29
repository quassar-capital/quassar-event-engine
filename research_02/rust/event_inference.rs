
#![forbid(unsafe_code)]

//! Research 02 — Independent Rust numerical reference kernels.
//!
//! Deterministic abnormal returns, cumulative abnormal returns,
//! and Holm multiple-testing adjustment.
//!
//! No causal interpretation or finite-sample inferential
//! guarantee is implied by these numerical operations.

use std::cmp::Ordering;

pub fn abnormal_returns(
    observed: &[f64],
    expected: &[f64],
) -> Result<Vec<f64>, String> {
    if observed.is_empty() || observed.len() != expected.len() {
        return Err("Input dimensions must match and be nonempty".into());
    }

    if observed.iter().chain(expected.iter()).any(|x| !x.is_finite()) {
        return Err("Inputs must be finite".into());
    }

    let result: Vec<f64> = observed
        .iter()
        .zip(expected.iter())
        .map(|(actual, baseline)| actual - baseline)
        .collect();

    if result.iter().any(|x| !x.is_finite()) {
        return Err("Abnormal-return overflow".into());
    }

    Ok(result)
}

pub fn cumulative_abnormal_return(
    abnormal: &[f64],
    start: usize,
    stop: usize,
) -> Result<f64, String> {
    if start >= stop || stop > abnormal.len() {
        return Err("Invalid half-open event window".into());
    }

    if abnormal.iter().any(|x| !x.is_finite()) {
        return Err("Abnormal returns must be finite".into());
    }

    let car: f64 = abnormal[start..stop].iter().sum();

    if !car.is_finite() {
        return Err("CAR calculation overflow".into());
    }

    Ok(car)
}

pub fn holm_adjustment(
    pvalues: &[f64],
) -> Result<Vec<f64>, String> {
    if pvalues.is_empty() {
        return Err("At least one p-value is required".into());
    }

    if pvalues.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0) {
        return Err("P-values must lie in [0, 1]".into());
    }

    let m = pvalues.len();
    let mut order: Vec<usize> = (0..m).collect();

    order.sort_by(|&a, &b| {
        pvalues[a]
            .partial_cmp(&pvalues[b])
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });

    let mut adjusted = vec![0.0; m];
    let mut running_max = 0.0_f64;

    for (rank, &index) in order.iter().enumerate() {
        let multiplier = (m - rank) as f64;
        let candidate = (pvalues[index] * multiplier).min(1.0);

        running_max = running_max.max(candidate);
        adjusted[index] = running_max;
    }

    Ok(adjusted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abnormal_returns_reference() {
        let actual = [0.03, -0.01, 0.04];
        let expected = [0.01, 0.00, 0.02];

        let result = abnormal_returns(&actual, &expected).unwrap();

        for (a, b) in result.iter().zip([0.02, -0.01, 0.02]) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn cumulative_return_reference() {
        let abnormal = [0.01, 0.02, -0.01, 0.03];

        let result =
            cumulative_abnormal_return(&abnormal, 1, 4).unwrap();

        assert!((result - 0.04).abs() < 1e-12);
    }

    #[test]
    fn holm_reference() {
        let pvalues = [0.01, 0.04, 0.03, 0.20];

        let adjusted = holm_adjustment(&pvalues).unwrap();
        let expected = [0.04, 0.09, 0.09, 0.20];

        for (a, b) in adjusted.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
