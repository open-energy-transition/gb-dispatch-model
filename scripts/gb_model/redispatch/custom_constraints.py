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
from snakemake.script import Snakemake

from scripts.gb_model._helpers import get_lines
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

    etys_boundaries_lines = snakemake.params.etys_boundaries_to_lines
    etys_boundaries_links = snakemake.params.etys_boundaries_to_links

    # Define Line-s and Link-p variable (power flow)
    line_s = n.model["Line-s"]
    link_p = n.model["Link-p"]

    for boundary in etys_capacities.index:
        # Get boundary capability
        capacity_mw = etys_capacities.loc[boundary]

        # Get all lines crossing the given boundary
        boundary_lines_mask = pd.Series(False, index=n.lines.index)
        for buses in etys_boundaries_lines.get(boundary, []):
            lines_mask = get_lines(n.lines, buses["bus0"], buses["bus1"])
            if not lines_mask.any():
                logger.warning(
                    f"No lines found for boundary '{boundary}' between "
                    f"buses '{buses['bus0']}' and '{buses['bus1']}'"
                )
            boundary_lines_mask = boundary_lines_mask | lines_mask

        if boundary_lines_mask.any():
            boundary_lines = n.lines[boundary_lines_mask].index

        # Get all DC links crossing the given boundary
        boundary_links_mask = pd.Series(False, index=n.links.index)

        for buses in etys_boundaries_links.get(boundary, []):
            links_mask = get_lines(n.links, buses["bus0"], buses["bus1"])

            # Filter only DC links
            dc_links_mask = n.links.carrier == "DC"
            combined_mask = links_mask & dc_links_mask

            if not combined_mask.any():
                logger.warning(
                    f"No DC links found for boundary '{boundary}' between "
                    f"buses '{buses['bus0']}' and '{buses['bus1']}'"
                )
            boundary_links_mask = boundary_links_mask | combined_mask

        boundary_links = n.links[boundary_links_mask].index

        if boundary_lines.empty:
            logger.warning(
                f"No lines found for boundary '{boundary}'. "
                f"Cannot apply ETYS constraint. Check configuration."
            )

        logger.info(
            f"Boundary {boundary}: {len(boundary_lines)} lines, {len(boundary_links)} DC links, "
            f"capacity={capacity_mw} MW"
        )

        # Get Line-s and Link-p for boundary lines and links
        line_s_boundary = line_s.sel(snapshot=snapshots, name=boundary_lines)
        link_p_boundary = link_p.sel(snapshot=snapshots, name=boundary_links)

        # Sum across lines and DC links to get total flow at the boundary
        lhs = line_s_boundary.sum("name") + link_p_boundary.sum("name")

        # Add bidirectional constraint: total flow ≤ boundary capability
        n.model.add_constraints(
            lhs <= capacity_mw,
            name=f"GlobalConstraint-etys_boundary_{boundary}_forward",
        )
        n.model.add_constraints(
            lhs >= -capacity_mw,
            name=f"GlobalConstraint-etys_boundary_{boundary}_backward",
        )

        logger.info(
            f"Added boundary constraint: -{capacity_mw:.0f} <= '{boundary}' <= {capacity_mw:.0f} MW"
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
