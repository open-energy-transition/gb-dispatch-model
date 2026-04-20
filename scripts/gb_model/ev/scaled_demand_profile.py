# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Scale EV electricity load profiles using annual and peak demand data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def scale_shape_with_annual_and_peak_data(
    demand_shape: pd.DataFrame,
    demand_annual: pd.DataFrame,
    demand_peak: pd.DataFrame,
    year: int,
    scaling_params: dict[str, float],
) -> pd.DataFrame:
    """
    Estimate the EV demand profile for the given year using shape, annual and peak data.

    Parameters
    ----------
    demand_shape : pd.DataFrame
        EV demand shape for each bus
    demand_annual : pd.DataFrame
        Annual demand data for each bus and year
    demand_peak : pd.DataFrame
        Peak demand data for each bus and year
    year : int
        Year used in the modelling
    scaling_params: dict[str, float]
        Dictionary containing EV profile adjustment parameters

    Returns
    -------
    pd.DataFrame
        Estimated EV demand profile
    """
    # Load the files

    # Select data for given year
    demand_annual = demand_annual.xs(year, level="year")
    demand_peak = demand_peak.xs(year, level="year")

    # Affine transformation to scale the maximum with peak and total energy with annual demand
    new_demand_shapes = {}
    for bus in demand_shape.columns:
        if bus not in demand_annual.index or bus not in demand_peak.index:
            continue

        peak = demand_peak.loc[bus, "p_nom"]
        annual = demand_annual.loc[bus, "p_set"]

        # Use shape-adjusting transformation to satisfy both peak and annual constraints
        new_demand_shapes[bus] = _adjust_profile_shape(
            demand_shape[bus],
            peak,
            annual,
            relative_peak_tolerance=scaling_params["relative_peak_tolerance"],
            relative_energy_tolerance=scaling_params["relative_energy_tolerance"],
            upper_optimization_bound=scaling_params["upper_optimization_bound"],
            lower_optimization_bound=scaling_params["lower_optimization_bound"],
        )

    logger.info(
        "EV demand profile successfully generated with both peak and annual constraints satisfied."
    )
    new_demand_shape_df = pd.DataFrame(new_demand_shapes)
    return new_demand_shape_df


def _adjust_profile_shape(
    shape_series: pd.Series,
    peak_target: float,
    annual_target: float,
    relative_peak_tolerance: float,
    relative_energy_tolerance: float,
    upper_optimization_bound: float,
    lower_optimization_bound: float,
) -> pd.Series:
    """
    Transform demand profile to match both peak and annual targets by adjusting the shape.

    This function can squeeze (make peakier) or widen (make flatter) the profile
    to satisfy both constraints simultaneously.

    Parameters
    ----------
    shape_series : pd.Series
        Normalized EV demand shape (sum = 1)
    peak_target : float
        Target peak demand (MW)
    annual_target : float
        Target annual energy (MWh)

    Returns
    -------
    pd.Series
        Transformed profile satisfying both constraints
    """
    # Normalize input to ensure sum = 1
    normalized_shape = _normalize(shape_series)

    # Try simple scaling first
    simple_scaled = normalized_shape * annual_target
    scaled_peak = simple_scaled.max()

    # If simple scaling satisfies peak constraint, return it
    if np.isclose(scaled_peak, peak_target, rtol=relative_peak_tolerance, atol=0):
        return simple_scaled

    # Need to adjust the shape - use power transformation
    # Higher gamma = more peaked, lower gamma = flatter

    def objective_function(gamma):
        """Objective function to find optimal shape parameter."""
        if gamma <= 0:
            return float("inf")

        # Apply power transformation and scale to match annual target
        scaled = _normalize(normalized_shape**gamma) * annual_target

        # Check how well we satisfy both constraints
        peak_error = abs(scaled.max() - peak_target) / peak_target
        annual_error = abs(scaled.sum() - annual_target) / annual_target

        return peak_error + annual_error

    result = minimize_scalar(
        objective_function,
        bounds=(lower_optimization_bound, upper_optimization_bound),
        method="bounded",
    )
    optimal_gamma = result.x

    # Apply optimal transformation
    final_profile = _normalize(normalized_shape**optimal_gamma) * annual_target

    # Verify constraints
    final_peak = final_profile.max()
    final_annual = final_profile.sum()

    logger.info(
        "Gamma optimization result for bus %s: optimal_gamma=%.4f, final_peak=%.2f MW, target_peak=%.2f MW, final_annual=%.2f MWh, target_annual=%.2f MWh",
        shape_series.name,
        optimal_gamma,
        final_peak,
        peak_target,
        final_annual,
        annual_target,
    )

    assert np.isclose(final_peak, peak_target, rtol=relative_peak_tolerance, atol=0), (
        f"Peak constraint violated after optimization for bus {shape_series.name} - "
        f"Expected: {peak_target:.2f} MW, Obtained: {final_peak:.2f} MW"
    )

    assert np.isclose(
        final_annual, annual_target, rtol=relative_energy_tolerance, atol=0
    ), (
        f"Annual constraint violated after optimization for bus {shape_series.name} - "
        f"Expected: {annual_target:.2f} MWh, Obtained: {final_annual:.2f} MWh"
    )

    return final_profile


def _normalize(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series so that its sum equals 1."""
    normalized = series / series.sum()
    return normalized


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            Path(__file__).stem,
            demand_type="baseline_electricity",
            year="2022",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    log_suffix = "-" + "_".join(snakemake.wildcards) if snakemake.wildcards else ""
    logger = logging.getLogger(Path(__file__).stem + log_suffix)

    gb_demand_annual = pd.read_csv(
        snakemake.input.gb_demand_annual, index_col=["bus", "year"]
    )
    eur_demand_annual = pd.read_csv(
        snakemake.input.eur_demand_annual, index_col=["load_type", "bus", "year"]
    ).xs("ev", level="load_type")
    all_demand_annual = pd.concat([gb_demand_annual, eur_demand_annual])
    profile_shape = pd.read_csv(
        snakemake.input.demand_shape, index_col=[0], parse_dates=True
    )
    gb_demand_peak = pd.read_csv(
        snakemake.input.gb_demand_peak, index_col=["bus", "year"]
    )
    eur_demand_peak = (
        (gb_demand_peak.p_nom / gb_demand_annual.p_set)
        .groupby("year")
        .mean()
        .mul(eur_demand_annual.p_set)
        .to_frame("p_nom")
    )
    all_demand_peak = pd.concat([gb_demand_peak, eur_demand_peak])
    load_profile = scale_shape_with_annual_and_peak_data(
        demand_annual=all_demand_annual,
        demand_peak=all_demand_peak,
        demand_shape=profile_shape,
        year=int(snakemake.wildcards.year),
        scaling_params=snakemake.params.scaling_params,
    )

    load_profile.to_csv(snakemake.output.csv)
