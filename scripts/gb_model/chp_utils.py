# SPDX-FileCopyrightText: gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT

"""
Utilities for simplified CHP modeling in the GB market model.

This module provides functions to add simplified CHP constraints to generators
based on heat demand profiles, without full sector-coupling.
"""

import logging

import pandas as pd
import pypsa
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
    shutdown_threshold: float = 0.1,
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
        - Index: snapshots (datetime)
        - Columns: bus names
        - Values: minimum operation level (0-1)

    Notes
    -----
    The heat-to-power ratio (c_b) represents the back-pressure characteristic:
    - For gas CHPs, typically 1.0-2.0
    - Higher values mean more heat per unit of electricity
    - This determines how much CHP must run to meet heat demand
    """
    try:
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
                index=pd.DatetimeIndex(ds.snapshots.values),
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
        # Avoid division by zero
        peak_demand = heat_demand_df.max()
        peak_demand[peak_demand == 0] = 1.0
        heat_profile = heat_demand_df / peak_demand

        # Convert heat demand to minimum power requirement
        # CHPs must run to meet heat demand: P_el = P_heat / heat_to_power_ratio
        # Then normalize to get p_min_pu as fraction of CHP capacity
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

    except FileNotFoundError:
        logger.error(f"Heat demand file not found: {heat_demand_path}")
        raise
    except Exception as e:
        logger.error(f"Error calculating CHP minimum operation: {e}")
        raise


def attach_chp_constraints(
    n: pypsa.Network,
    powerplants: pd.DataFrame,
    heat_demand_path: str,
    heat_to_power_ratio: float,
    min_operation_level: float,
    shutdown_threshold: float = 0.1,
) -> None:
    """
    Attach simplified CHP constraints to generators in the network.

    Identifies CHP generators and applies minimum operation constraints
    based on heat demand profiles.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object. Generators must have a 'set' attribute
        for CHP identification (values: 'CHP', 'PP', 'Store').
    powerplants : pd.DataFrame
        Powerplants dataframe including CHP identification.
    heat_demand_path : str
        Path to hourly heat demand NetCDF file.
    heat_to_power_ratio : float
        Heat-to-power ratio for CHPs (c_b coefficient).
    min_operation_level : float
        Minimum operation level (0-1) when CHP is running.
    shutdown_threshold : float, optional
        Heat demand threshold below which CHPs can shut down.
        Default is 0.1 (10% of peak demand).

    Returns
    -------
    None
        Modifies the network object in-place.

    Notes
    -----
    This is a simplified CHP model that:
    - Models CHPs as generators (not Links)
    - Uses p_min_pu to enforce minimum operation following heat demand
    - Does not model heat buses or sector coupling
    - Suitable for electricity-only optimization with CHP constraints
    - Requires generators to have 'set' attribute for precise CHP identification

    Raises
    ------
    ValueError
        If generators do not have 'set' attribute
    FileNotFoundError
        If heat demand file is not found
    KeyError
        If required configuration keys are missing
    """
    # Identify CHP powerplants
    chp_plants = identify_chp_powerplants(powerplants)

    if chp_plants.empty:
        logger.info("No CHP powerplants to constrain.")
        return

    # Find corresponding generators in the network
    # CHPs are identified using the 'set' attribute
    if "set" not in n.generators.columns:
        logger.error(
            "'set' attribute not found in generators. "
            "This is required for CHP identification. "
            "Ensure attach_conventional_generators() properly sets this attribute."
        )
        raise ValueError("'set' attribute missing from generators dataframe")

    chp_generators = n.generators[n.generators["set"] == "CHP"]

    if chp_generators.empty:
        logger.info(
            "No CHP generators found in the network. "
            f"Total generators: {len(n.generators)}, "
            f"generators by set: {n.generators.groupby('set').size().to_dict()}"
        )
        return

    logger.info(
        f"Applying CHP constraints to {len(chp_generators)} generators "
        f"with total capacity {chp_generators.p_nom.sum():.1f} MW"
    )

    # Calculate minimum operation profile
    # Get buses from matched CHP generators
    buses = pd.Index(chp_generators["bus"].unique())

    p_min_pu = calculate_chp_minimum_operation(
        heat_demand_path=heat_demand_path,
        buses=buses,
        heat_to_power_ratio=heat_to_power_ratio,
        min_operation_level=min_operation_level,
        shutdown_threshold=shutdown_threshold,
    )

    # Map minimum operation to generators (vectorized)
    # Each generator inherits the profile of its bus
    gen_to_bus = chp_generators["bus"]

    # Filter to only generators with available heat demand data
    valid_gens = gen_to_bus[gen_to_bus.isin(p_min_pu.columns)]
    missing_gens = gen_to_bus[~gen_to_bus.isin(p_min_pu.columns)]

    if not missing_gens.empty:
        logger.warning(
            f"No heat demand data for {len(missing_gens)} generators at buses: {list(missing_gens.unique())}. "
            "These generators will have no CHP constraint."
        )

    # Vectorized assignment: rename p_min_pu columns from bus names to generator indices
    # Select columns for each generator's bus, then rename to generator index
    p_min_pu_for_gens = p_min_pu[valid_gens.values].copy()
    p_min_pu_for_gens.columns = valid_gens.index

    # Assign all generators at once
    n.generators_t.p_min_pu[valid_gens.index] = p_min_pu_for_gens

    # Log summary statistics
    avg_min = p_min_pu.mean().mean()
    max_min = p_min_pu.max().max()

    logger.info(
        f"CHP constraints applied successfully. "
        f"Average minimum operation: {avg_min:.1%}, "
        f"Maximum minimum operation: {max_min:.1%}"
    )
