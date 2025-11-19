# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Scale the boundary capabilities in a GB model network to match ETYS capacities.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
from scipy.optimize import minimize

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def _get_lines(lines: pd.DataFrame, bus1: str, bus0: str) -> pd.Series:
    all_lines = set(lines["bus0"]).union(set(lines["bus1"]))
    for bus in (bus0, bus1):
        if "GB " + bus not in all_lines:
            logger.warning(f"Bus 'GB {bus}' not found in network lines")
    return ((lines["bus0"] == "GB " + bus0) & (lines["bus1"] == "GB " + bus1)) | (
        (lines["bus0"] == "GB " + bus1) & (lines["bus1"] == "GB " + bus0)
    )


def get_boundary_s_noms(
    network: pypsa.Network, etys_boundaries: dict[str, list[dict[str, str]]]
) -> dict[str, list[float]]:
    s_noms: dict[str, list[float]] = {}
    for boundary, bus_groups in etys_boundaries.items():
        s_noms[boundary] = []
        for buses in bus_groups:
            lines = _get_lines(network.lines, buses["bus0"], buses["bus1"])
            s_nom = network.lines[lines].s_nom.sum()
            if s_nom == 0:
                print(boundary, buses)
            s_noms[boundary].append(s_nom)
    return s_noms


def get_s_max_pu(
    network: pypsa.Network,
    s_noms: dict[str, list[float]],
    etys_caps: pd.DataFrame,
    etys_boundaries: dict[str, list[dict[str, str]]],
) -> pd.Series:
    merged_df = (
        pd.Series(s_noms)
        .apply(lambda x: sum(x) if x else float("nan"))
        .to_frame("s_nom")
        .merge(etys_caps, left_index=True, right_on="boundary_name")
        .set_index("boundary_name")
    )
    s_max_pu = merged_df["capability_mw"] / merged_df["s_nom"]

    # Assign min/max bounds for each line based on all boundaries it belongs to
    network.lines = network.lines.assign(s_max_pu_max=0, s_max_pu_min=float("inf"))
    for boundary, bus_groups in etys_boundaries.items():
        for buses in bus_groups:
            lines = _get_lines(network.lines, buses["bus0"], buses["bus1"])
            for lim in ["min", "max"]:
                network.lines.loc[lines, f"s_max_pu_{lim}"] = network.lines.loc[
                    lines, f"s_max_pu_{lim}"
                ].map(lambda x: getattr(np, lim)([x, s_max_pu[boundary]]))

    # Optimization: find s_max_pu for each line that minimizes error to target capabilities
    # Build mapping from lines to boundaries they contribute to
    all_lines = []
    for boundary, bus_groups in etys_boundaries.items():
        for buses in bus_groups:
            all_lines.append(_get_lines(network.lines, buses["bus0"], buses["bus1"]))
    relevant_lines = pd.concat(all_lines, axis=1).any(axis=1)

    if not relevant_lines.any():
        logger.warning("No lines found belonging to ETYS boundaries")
        return pd.Series(dtype=float)
    # Initial guess: midpoint of min/max range
    x0 = (
        network.lines.loc[relevant_lines]
        .apply(lambda row: (row["s_max_pu_min"] + row["s_max_pu_max"]) / 2, axis=1)
        .values
    )

    # Bounds for each line
    bounds = [
        network.lines.loc[relevant_lines, ["s_max_pu_min", "s_max_pu_max"]].values
    ]

    # Objective function: sum of squared errors between target and achieved boundary capacities
    def objective(x):
        # Apply s_max_pu values to network temporarily
        network.lines.loc[relevant_lines, "s_max_pu"] = x

        # Recalculate boundary s_noms with updated s_max_pu
        current_s_noms = get_boundary_s_noms(network, etys_boundaries)
        boundary_totals = pd.Series(current_s_noms).apply(
            lambda vals: sum(vals) if vals else 0
        )
        df_ = boundary_totals.to_frame("s_nom").merge(
            etys_caps, left_index=True, right_on="boundary_name"
        )
        # Calculate error for each boundary
        total_error = sum(abs(df_.s_nom - df_.capability_mw))

        return total_error

    # Optimize
    logger.info(
        f"Optimizing s_max_pu for {relevant_lines.sum()} lines across {len(etys_boundaries)} boundaries"
    )
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds[0],
        options={"maxiter": 10000, "ftol": 1e-6},
    )

    if not result.success:
        logger.warning(f"Optimization did not fully converge: {result.message}")
    else:
        logger.info(
            f"Optimization converged successfully with final error: {result.fun:.2f} MW²"
        )

    # Create series with optimized s_max_pu values
    s_max_pu_optimised = pd.Series(result.x, index=relevant_lines[relevant_lines].index)

    # Log the results
    network.lines.loc[relevant_lines, "s_max_pu"] = result.x
    final_s_noms = get_boundary_s_noms(network, etys_boundaries)
    final_totals = pd.Series(final_s_noms).apply(lambda vals: sum(vals) if vals else 0)

    for boundary in etys_boundaries.keys():
        scaled_capacity = final_totals.get(boundary, 0)
        target_capacity = merged_df.loc[boundary, "capability_mw"]
        error_pct = 100 * (scaled_capacity - target_capacity) / target_capacity
        logger.info(
            f"{boundary}: target={target_capacity:.0f} MW, achieved={scaled_capacity:.0f} MW, "
            f"error={error_pct:+.2f}%"
        )

    return s_max_pu_optimised


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    network = pypsa.Network(snakemake.input.network)
    etys_boundaries = snakemake.params.etys_boundaries_to_lines
    etys_caps = pd.read_csv(snakemake.input.etys_caps)

    boundary_s_noms = get_boundary_s_noms(network, etys_boundaries)

    s_max_pu_optimised = get_s_max_pu(
        network, boundary_s_noms, etys_caps, etys_boundaries
    )

    s_max_pu_optimised.to_csv(snakemake.output.csv)
