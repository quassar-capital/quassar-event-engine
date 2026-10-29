
"""Reproducible scientific calibration for Research 02.

Evaluates the existing Max-T inference procedure under
complete-null and partial-null Gaussian AR(1) designs.

Outputs descriptive Monte Carlo evidence, not universal
finite-sample guarantees.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from research_02.python.fwer_experiment import (
    FWERConfig,
    run_fwer_experiment,
)
from research_02.python.partial_null_experiment import (
    PartialNullConfig,
    run_partial_null_experiment,
)


def _scenario(
    name: str,
    result: dict,
) -> dict:
    """Attach an explicit scenario identifier."""
    return {
        "scenario": name,
        **result,
    }


def run_calibration() -> dict:
    """Run a fixed, reproducible calibration design."""

    common = {
        "n_observations": 120,
        "n_instruments": 8,
        "n_simulations": 300,
        "n_resamples": 299,
        "block_length": 6,
        "alpha": 0.05,
        "seed": 2026,
    }

    designs = [
        (
            "complete_null_independent",
            FWERConfig(
                **common,
                temporal_correlation=0.0,
                cross_correlation=0.0,
            ),
        ),
        (
            "complete_null_moderate_dependence",
            FWERConfig(
                **common,
                temporal_correlation=0.3,
                cross_correlation=0.4,
            ),
        ),
        (
            "complete_null_strong_dependence",
            FWERConfig(
                **common,
                temporal_correlation=0.7,
                cross_correlation=0.8,
            ),
        ),
    ]

    complete_null_results = [
        _scenario(
            name,
            run_fwer_experiment(config),
        )
        for name, config in designs
    ]

    partial_config = PartialNullConfig(
        **common,
        n_alternatives=3,
        effect_size=0.5,
        temporal_correlation=0.3,
        cross_correlation=0.4,
    )

    partial_result = _scenario(
        "partial_null_moderate_dependence",
        run_partial_null_experiment(
            partial_config
        ),
    )

    return {
        "research": (
            "02 - High-Dimensional Event Study Inference"
        ),
        "experiment": (
            "Max-T HAC block-bootstrap calibration"
        ),
        "design_family": (
            "stationary Gaussian AR(1) panels "
            "with equicorrelated innovations"
        ),
        "inference_method": (
            "single-step Max-T circular block bootstrap"
        ),
        "studentization": "Bartlett HAC",
        "nominal_alpha": 0.05,
        "complete_null": complete_null_results,
        "partial_null": partial_result,
        "interpretation": {
            "complete_null": (
                "Estimated weak FWER under the "
                "specified complete-null designs."
            ),
            "partial_null": (
                "Estimated false-family rejection "
                "probability among true nulls, "
                "with alternatives present."
            ),
            "uncertainty": (
                "Reported Monte Carlo standard errors "
                "describe simulation uncertainty."
            ),
            "limitations": [
                "Finite simulation counts.",
                "Finite bootstrap resample counts.",
                "Gaussian AR(1) data-generating models.",
                "Equicorrelated cross-sectional innovations.",
                "Fixed HAC bandwidth and block length.",
                "No universal FWER guarantee.",
                "No causal identification.",
            ],
        },
    }


def main() -> None:
    """Execute and save deterministic calibration results."""

    results = run_calibration()

    destination = Path(
        "research_02/results/calibration.json"
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            results,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            results,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )

    print(
        f"\nCalibration results saved to {destination}"
    )


if __name__ == "__main__":
    main()
