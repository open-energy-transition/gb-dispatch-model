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
    etys_capacities = pd.read_csv(snakemake.input.etys_caps, index_col="boundary_name")
    etys_boundaries_lines = snakemake.params.etys_boundaries_to_lines
    etys_boundaries_links = snakemake.params.etys_boundaries_to_links

    # Define Line-s and Link-p variable (power flow)
    line_s = n.model["Line-s"]
    link_p = n.model["Link-p"]

    for boundary in etys_capacities.index:
        # Get boundary capability
        capacity_mw = etys_capacities.loc[boundary, "capability_mw"]

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
        line_s_boundary = line_s.sel(snapshot=snapshots, Line=boundary_lines)
        link_p_boundary = link_p.sel(snapshot=snapshots, Link=boundary_links)

        # Sum across lines and DC links to get total flow at the boundary
        lhs = line_s_boundary.sum("Line") + link_p_boundary.sum("Link")

        # Add bidirectional constraint: total flow ≤ boundary capability
        n.model.add_constraints(
            lhs <= capacity_mw, name=f"etys_boundary_{boundary}_forward"
        )
        n.model.add_constraints(
            lhs >= -capacity_mw, name=f"etys_boundary_{boundary}_backward"
        )

        logger.info(
            f"Added boundary constraint: -{capacity_mw:.0f} <= '{boundary}' <= {capacity_mw:.0f} MW"
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
