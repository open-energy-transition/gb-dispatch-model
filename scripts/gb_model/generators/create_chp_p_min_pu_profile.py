# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Utilities for simplified CHP modeling in the GB market model.

This module provides functions to add simplified CHP constraints to generators
based on heat demand profiles, without full sector-coupling.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def identify_chp_powerplants(powerplants: pd.DataFrame) -> pd.DataFrame:
    """
    Identify CHP powerplants from the powerplants dataframe.

    CHPs are identified by the 'Set' column containing 'CHP'.

    Parameters
    ----------
    powerplants : pd.DataFrame
        Powerplants dataframe with columns including 'set' (lowercase after processing).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only CHP powerplants.
    """
    if "set" not in powerplants.columns:
        logger.warning(
            "Column 'set' not found in powerplants data. No CHPs identified."
        )
        return pd.DataFrame()

    chp_plants = powerplants[powerplants["set"] == "CHP"].copy()

    if chp_plants.empty:
        logger.warning("No CHP powerplants found in the dataset.")
    else:
        logger.info(
            f"Identified {len(chp_plants)} CHP powerplants with total capacity "
            f"{chp_plants['p_nom'].sum():.1f} MW"
        )

    return chp_plants


def calculate_chp_minimum_operation(
    heat_demand_path: str,
    buses: pd.Index,
    heat_to_power_ratio: float,
    min_operation_level: float,
    shutdown_threshold: float,
) -> pd.DataFrame:
    """
    Calculate minimum operation profile (p_min_pu) for CHPs based on heat demand.

    The function:
    1. Loads total heat demand for each bus
    2. Normalizes heat demand to peak demand per bus
    3. Converts to electricity requirement using heat-to-power ratio
    4. Applies minimum operation level floor

    Parameters
    ----------
    heat_demand_path : str
        Path to hourly heat demand NetCDF file.
    buses : pd.Index
        Index of buses in the network.
    heat_to_power_ratio : float
        Ratio of heat output to electrical output (c_b coefficient).
        E.g., 1.5 means CHPs produce 1.5x more heat than electricity.
    min_operation_level : float
        Minimum operation level as fraction of capacity (0-1).
        E.g., 0.3 means 30% of capacity.
    shutdown_threshold : float, optional
        Heat demand threshold (as fraction of peak demand) below which CHPs can shut down.
        Default is 0.1 (10% of peak demand).

    Returns
    -------
    pd.DataFrame
        Time series of p_min_pu values with:
        - Index: snapshot (datetime)
        - Columns: bus names
        - Values: minimum operation level (0-1)

    Notes
    -----
    The heat-to-power ratio (c_b) represents the back-pressure characteristic:
    - For gas CHPs, typically 1.0-2.0
    - Higher values mean more heat per unit of electricity
    - This determines how much CHP must run to meet heat demand
    """

    # Load heat demand dataset
    ds = xr.open_dataset(heat_demand_path)

    # Sum all heat demand types (residential + services, water + space)
    heat_vars = [
        v
        for v in ds.data_vars
        if v
        in [
            "residential water",
            "residential space",
            "services water",
            "services space",
        ]
    ]

    if not heat_vars:
        logger.warning(
            f"No heat demand variables found in {heat_demand_path}. "
            "Available variables: " + ", ".join(ds.data_vars)
        )
        # Return zero minimum operation
        return pd.DataFrame(
            0.0,
            index=ds.snapshots.to_index(),
            columns=buses,
        )

    # Sum all heat demand types
    total_heat_demand = sum(ds[var] for var in heat_vars)

    # Convert to pandas DataFrame
    heat_demand_df = total_heat_demand.to_pandas()

    # Rename columns to match bus names (remove any prefixes if needed)
    # Filter to only buses that exist in the network
    heat_demand_df = heat_demand_df.reindex(columns=buses, fill_value=0.0)

    # Normalize by peak demand per bus to get profile (0-1)
    # Replace division by zero with zero
    heat_profile = heat_demand_df.divide(heat_demand_df.max()).replace(float("nan"), 0)

    # Convert heat demand to minimum power requirement
    # CHPs must run to meet heat demand: P_el = P_heat / heat_to_power_ratio
    p_min_pu = heat_profile / heat_to_power_ratio

    # Apply minimum operation level floor
    # CHPs cannot operate below this threshold when they run
    # Clip to ensure p_min_pu is at least min_operation_level when heat demand exists
    p_min_pu = p_min_pu.clip(lower=min_operation_level)
    # Where heat demand is very low, allow CHP to shut down completely
    # (e.g., below shutdown_threshold fraction of peak demand, CHP can be off)
    p_min_pu[heat_profile < shutdown_threshold] = 0.0

    logger.info(
        f"Calculated CHP minimum operation profile: "
        f"mean={p_min_pu.mean().mean():.3f}, "
        f"max={p_min_pu.max().max():.3f}"
    )

    return p_min_pu


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)

    buses = gpd.read_file(snakemake.input.regions)["name"]
    heat_demand_path = snakemake.input.heat_demand
    heat_to_power_ratio = snakemake.params.heat_to_power_ratio
    min_operation_level = snakemake.params.min_operation_level
    shutdown_threshold = snakemake.params.shutdown_threshold

    p_min_pu = calculate_chp_minimum_operation(
        heat_demand_path=heat_demand_path,
        buses=buses,
        heat_to_power_ratio=heat_to_power_ratio,
        min_operation_level=min_operation_level,
        shutdown_threshold=shutdown_threshold,
    )

    # PyPSA expects the index of timeseries data to be named 'snapshot'
    p_min_pu.rename_axis(index="snapshot").to_csv(snakemake.output.chp_p_min_pu)
