# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Custom constraints provider.

Defines custom constraints for the GB model.
"""

import logging

import pandas as pd
import pypsa
import xarray as xr
from linopy import merge
from snakemake.script import Snakemake

from scripts.gb_model.dispatch.custom_constraints import remove_KVL_constraints

logger = logging.getLogger(__name__)


def set_boundary_constraints(
    n: pypsa.Network,
    snapshots: pd.Index,
    snakemake: Snakemake,
) -> None:
    """
    Limit new line flows across each boundary to satisfy ETYS boundary capabilities.

    Args:
        n (pypsa.Network): The PyPSA network to add constraints.
        snapshots (pd.Index): The snapshots of the network.
        snakemake (snakemake.Snakemake): The snakemake object for parameters and config.
    """
    # Load ETYS capacities
    etys_capacities = pd.read_csv(
        snakemake.input.current_etys_caps, index_col="boundary_name"
    ).capability_mw

    # If future ETYS caps are provided, use them instead of initial levels,
    # gap filling with initial levels where future caps are missing.
    if snakemake.input.future_etys_caps:
        year = int(snakemake.wildcards.year)
        future_caps = pd.read_csv(
            snakemake.input.future_etys_caps, index_col=["boundary_name", "year"]
        ).capability_mw.xs(year, level="year")
        manual_caps = pd.DataFrame(snakemake.params.manual_future_etys_caps).loc[year]
        etys_capacities_all_boundaries = pd.concat([future_caps, manual_caps]).reindex(
            etys_capacities.index
        )
        if (isna := etys_capacities_all_boundaries.isna()).any():
            logger.warning(
                f"Future ETYS capacities are missing for some boundaries: "
                f"{etys_capacities_all_boundaries[isna].index.tolist()}. \n"
                "Filling missing values with current capacities."
            )
        etys_capacities = etys_capacities_all_boundaries.fillna(etys_capacities)

    etys_boundary_crossings = pd.read_csv(snakemake.input.boundary_crossings)

    # Define Line-s and Link-p variable (power flow)
    line_s = n.model["Line-s"]
    link_p = n.model["Link-p"]

    lhs_exprs = []

    boundary_index = pd.Index(etys_capacities.index, name="boundary")
    for boundary in boundary_index:
        # Get boundary capability
        capacity_mw = etys_capacities.loc[boundary]

        boundary_lines_mask = etys_boundary_crossings.query(
            "Boundary_n == @boundary and component == 'Line'"
        )
        boundary_links_mask = etys_boundary_crossings.query(
            "Boundary_n == @boundary and component == 'Link'"
        )

        if boundary_lines_mask.empty and boundary_links_mask.empty:
            logger.warning(
                f"No lines or links found for boundary '{boundary}'. "
                f"Cannot apply ETYS constraint. Check configuration."
            )
            continue

        logger.info(
            f"Boundary {boundary}: {len(boundary_lines_mask)} lines, {len(boundary_links_mask)} DC links, "
            f"capacity={capacity_mw} MW"
        )
        boundary_lines = boundary_lines_mask["name"].tolist()
        boundary_links = boundary_links_mask["name"].tolist()
        line_s_boundary = line_s.sel(snapshot=snapshots, name=boundary_lines)
        link_p_boundary = link_p.sel(snapshot=snapshots, name=boundary_links)

        # Sum across lines and DC links to get total flow at the boundary
        lhs = line_s_boundary.sum("name") + link_p_boundary.sum("name")
        lhs_exprs.append(lhs)

    lhs_merged = merge(lhs_exprs, dim="boundary").assign_coords(boundary=boundary_index)

    bounds = xr.DataArray(
        etys_capacities.values,
        coords=[boundary_index],
    )

    n.model.add_constraints(lhs_merged <= bounds, name="etys_boundary_forward")
    n.model.add_constraints(lhs_merged >= -bounds, name="etys_boundary_backward")

    logger.info(
        f"Added {len(boundary_index)} boundary constraints with explicit 'boundary' dimension"
    )


def update_storage_p_bounds(n: pypsa.Network) -> None:
    """
    Update the bounds of storage unit dispatch and store power to make up for insufficient default bounding.

    Args:
        n (pypsa.Network): The PyPSA network to update.
    """
    # `p_dispatch`/`p_store` have correct upper bounds but no lower, so we copy one from the other to _fix_ the storage unit flows.
    for direction in ["dispatch", "store"]:
        n.model.constraints[
            f"StorageUnit-fix-p_{direction}-lower"
        ].rhs = n.model.constraints[f"StorageUnit-fix-p_{direction}-upper"].rhs


def update_storage_balance(n: pypsa.Network) -> None:
    """
    Update the energy balance mathematics to include up and down ramping

    Args:
        n (pypsa.Network): The PyPSA network to update.
    """
    storage_unit_balance = n.model.constraints["StorageUnit-energy_balance"]
    ramp_names = n.generators[
        n.generators.carrier.str.startswith("StorageUnit ramp")
    ].index
    ramp_p = n.model["Generator-p"].sel(name=ramp_names)
    idx = ramp_p.coords["name"].str.replace(r" ramp (up|down)", "")
    ramp_p_grouped = ramp_p.groupby(idx).sum()

    # ramping up and ramping down `p` already have the correct signs (positive and negative, respectively),
    # so no need to invert one of them when applying to the LHS.
    # `ramp_up` is equivalent to `p_dispatch`, `ramp_down` is equivalent to `p_store`
    storage_unit_balance.lhs -= ramp_p_grouped.sel(storage_unit_balance.coords)
    # Log a randomly selected snapshot for verification
    logger.info(
        f"Updated energy balance math for storage units. Example: {storage_unit_balance.isel(snapshot=10)}"
    )


def custom_constraints(
    n: pypsa.Network,
    snapshots: pd.Index,
    snakemake: Snakemake,
) -> None:
    """
    Apply custom constraints to the PyPSA network.

    Args:
        n (pypsa.Network): The PyPSA network to modify.
        snapshots (pd.Index): The snapshots of the network.
        snakemake (snakemake.Snakemake): The snakemake object for parameters and config.
    """
    # Apply boundary constraints
    set_boundary_constraints(n, snapshots, snakemake)
    remove_KVL_constraints(n, snapshots, snakemake)
    update_storage_p_bounds(n)
    update_storage_balance(n)
