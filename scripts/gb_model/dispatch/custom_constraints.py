# SPDX-FileCopyrightText: : gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT
import logging

import pandas as pd
import pypsa
from snakemake.script import Snakemake

logger = logging.getLogger(__name__)


def remove_KVL_constraints(n, snapshots, snakemake):
    """
    Add custom extra functionality constraints to remove KVL constraints.
    """
    n.model.remove_constraints("Kirchhoff-Voltage-Law")
    logger.info("Remove KVL constraints from the model")


def limit_annual_nuclear_operation(n, snapshots, snakemake):
    """
    Add custom extra functionality constraints to limit annual nuclear capacity factor to within a specified range.
    """
    # Example implementation (details would depend on specific requirements)
    max_annual_cf = snakemake.params.nuclear_max_annual_capacity_factor
    min_annual_cf = snakemake.params.nuclear_min_annual_capacity_factor
    nuclear_generators = n.generators[n.generators.carrier == "nuclear"].index
    total_generation = (
        n.model["Generator-p"].sel(name=nuclear_generators)
        * n.snapshot_weightings["generators"].to_xarray()
    ).sum("snapshot")
    max_generation = (
        n.generators.loc[nuclear_generators, "p_nom"]
        .rename_axis(index="name")
        .to_xarray()
        * n.snapshot_weightings["generators"].to_xarray()
    ).sum("snapshot")
    n.model.add_constraints(
        lhs=total_generation,
        sign="<=",
        rhs=max_annual_cf * max_generation,
        name="GlobalConstraint-max_annual_nuclear_operation",
    )
    n.model.add_constraints(
        lhs=total_generation,
        sign=">=",
        rhs=min_annual_cf * max_generation,
        name="GlobalConstraint-min_annual_nuclear_operation",
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
    remove_KVL_constraints(n, snapshots, snakemake)
    limit_annual_nuclear_operation(n, snapshots, snakemake)
