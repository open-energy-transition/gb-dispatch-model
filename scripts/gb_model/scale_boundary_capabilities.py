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
import plotly.express as px
import plotly.graph_objects as go
import pypsa
from scipy.optimize import minimize

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def _get_lines(lines: pd.DataFrame, bus1: str | int, bus0: str | int) -> pd.Series:
    """
    Get a boolean mask of lines connecting bus0 and bus1.
    This accounts for cases where bus0 and bus1 may be swapped.
    """
    all_lines = set(lines["bus0"]).union(set(lines["bus1"]))
    for bus in (bus0, bus1):
        if f"GB {bus}" not in all_lines:
            logger.warning(f"Bus 'GB {bus}' not found in network lines")
    return ((lines["bus0"] == f"GB {bus0}") & (lines["bus1"] == f"GB {bus1}")) | (
        (lines["bus0"] == f"GB {bus1}") & (lines["bus1"] == f"GB {bus0}")
    )


def get_boundary_s_noms(
    lines: pd.DataFrame,
    etys_boundaries: dict[str, list[dict[str, str]]],
    s_max_pu: pd.Series | int = 1,
) -> pd.Series:
    """
    Calculate total boundary `s_nom` by summing all lines crossing each boundary.

    Args:
        lines (pd.DataFrame): PyPSA network lines DataFrame
        etys_boundaries (dict[str, list[dict[str, str]]]): PyPSA bus to ETYS boundaries mapping

    Returns:
        pd.Series: boundary name to total s_nom mapping
    """
    s_noms: dict[str, list[float]] = {}
    for boundary, bus_groups in etys_boundaries.items():
        s_noms[boundary] = []
        for buses in bus_groups:
            lines_mask = _get_lines(lines, buses["bus0"], buses["bus1"])
            s_nom = lines.s_nom.mul(s_max_pu)[lines_mask].sum()
            if s_nom == 0:
                logger.warning(
                    f"No lines found for boundary '{boundary}' between "
                    f"buses '{buses['bus0']}' and '{buses['bus1']}'"
                )
            s_noms[boundary].append(s_nom)
    s_noms_df = pd.Series(s_noms).apply(lambda x: sum(x) if x else float("nan"))
    return s_noms_df


def _s_noms_compare(s_noms: pd.Series, etys_caps: pd.DataFrame) -> pd.DataFrame:
    merged_df = (
        pd.Series(s_noms)
        .to_frame("s_nom")
        .merge(etys_caps, left_index=True, right_on="boundary_name")
        .set_index("boundary_name")
    )
    return merged_df


def get_s_max_pu(
    network: pypsa.Network,
    s_noms: pd.Series,
    etys_caps: pd.DataFrame,
    etys_boundaries: dict[str, list[dict[str, str]]],
) -> pd.Series:
    """
    Get our best guess of scaling factors for `s_nom` for each PyPSA network line,
    based on the target of the ETYS boundary capabilities.

    Args:
        network (pypsa.Network): PyPSA network object
        s_noms (pd.Series): boundary name to total s_nom values
        etys_caps (pd.DataFrame): ETYS boundary capabilities DataFrame
        etys_boundaries (dict[str, list[dict[str, str]]]):  PyPSA bus to ETYS boundaries mapping

    Returns:
        pd.Series: Series of s_max_pu values for relevant lines
    """
    merged_df = _s_noms_compare(s_noms, etys_caps)
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

    # Optimisation: find s_max_pu for each line that minimizes error to target capabilities
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
    def _objective(x):
        # Recalculate boundary s_noms with updated s_max_pu
        boundary_totals = get_boundary_s_noms(
            network.lines.loc[relevant_lines], etys_boundaries, s_max_pu=x
        )
        df_ = _s_noms_compare(boundary_totals, etys_caps)
        # Calculate relative error for each boundary
        total_error = np.sqrt(
            sum((df_.s_nom - df_.capability_mw).div(df_.capability_mw) ** 2)
        )
        return total_error

    # Optimize
    logger.info(
        f"Optimizing s_max_pu for {relevant_lines.sum()} lines across {len(etys_boundaries)} boundaries"
    )
    result = minimize(_objective, x0, bounds=bounds[0])

    if not result.success:
        logger.warning(f"Optimisation did not fully converge: {result.message}")
    else:
        logger.info(
            f"Optimisation converged successfully with final error: {result.fun:.2f}"
        )

    # Create series with optimised s_max_pu values
    s_max_pu_optimised = pd.Series(result.x, index=relevant_lines[relevant_lines].index)

    # Log the results
    final_s_noms = get_boundary_s_noms(
        network.lines.loc[relevant_lines], etys_boundaries, s_max_pu=result.x
    )

    for boundary in etys_boundaries.keys():
        scaled_capacity = final_s_noms.get(boundary, 0)
        target_capacity = merged_df.loc[boundary, "capability_mw"]
        error_pct = 100 * (scaled_capacity - target_capacity) / target_capacity
        logger.info(
            f"{boundary}: target={target_capacity:.0f} MW, achieved={scaled_capacity:.0f} MW, "
            f"error={error_pct:+.2f}%"
        )

    return s_max_pu_optimised


def plot_compare_s_nom(
    s_noms: pd.Series, etys_caps: pd.DataFrame, s_noms_optimised: pd.Series
) -> go.Figure:
    df_original = _s_noms_compare(s_noms, etys_caps)
    df_scaled = _s_noms_compare(s_noms_optimised, etys_caps)
    df_plot = pd.concat(
        [df_original.assign(type="Original"), df_scaled.assign(type="Scaled")]
    ).reset_index()
    max_val = df_plot[["s_nom", "capability_mw"]].max().max()
    fig = px.scatter(
        df_plot, x="s_nom", y="capability_mw", text="boundary_name", color="type"
    )
    fig = fig.add_scatter(
        x=[0, max_val],
        y=[0, max_val],
        marker={"opacity": 0},
        opacity=0.5,
        showlegend=False,
    )
    fig = fig.update_layout(
        yaxis={
            "title": {"text": "ETYS boundary capability (MW)", "font": {"size": 20}}
        },
        xaxis={
            "title": {
                "text": "PyPSA-Eur boundary capability (MW)",
                "font": {"size": 20},
            }
        },
        font={"size": 15},
        width=1100,
        height=900,
    )
    fig = fig.update_traces(marker={"size": 10}, textposition="top center")
    return fig


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    network = pypsa.Network(snakemake.input.network)
    etys_boundaries = snakemake.params.etys_boundaries_to_lines
    etys_caps = pd.read_csv(snakemake.input.etys_caps)

    boundary_s_noms = get_boundary_s_noms(network.lines, etys_boundaries)

    s_max_pu_optimised = get_s_max_pu(
        network, boundary_s_noms, etys_caps, etys_boundaries
    )
    for line in snakemake.params.prune_lines:
        mask = _get_lines(network.lines, line["bus0"], line["bus1"])
        if mask.any():
            s_max_pu_optimised = pd.concat(
                [s_max_pu_optimised, pd.Series(0, index=mask[mask].index)]
            )
            logger.info(
                f"Pruned line between bus {line['bus0']} and bus {line['bus1']}"
            )
        else:
            logger.warning(
                f"No line found to prune between bus {line['bus0']} and bus {line['bus1']}"
            )
    fig = plot_compare_s_nom(
        get_boundary_s_noms(network.lines, etys_boundaries, network.lines.s_max_pu),
        etys_caps,
        get_boundary_s_noms(network.lines, etys_boundaries, s_max_pu_optimised),
    )
    fig.write_html(snakemake.output.html)

    s_max_pu_optimised.to_csv(snakemake.output.csv)
