# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Prepare network for constrained optimization.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

LARGE_NUMBER = 1e6


def _get_intra_gb(df: pd.DataFrame) -> pd.Series:
    return df.bus0.str.startswith("GB") & df.bus1.str.startswith("GB")


def copperplate_gb(network: pypsa.Network) -> None:
    """
    Modify the network in-place to represent a copperplate model of Great Britain by
    setting the capacities of all transmission lines to a very high value.

    Parameters
    ----------
    network : pypsa.Network
        The input PyPSA network representing the GB power system.

    Returns
    -------
    pypsa.Network
        The modified PyPSA network with unconstrained transmission lines.
    """
    network.lines.loc[_get_intra_gb(network.lines), "s_nom"] = LARGE_NUMBER
    network.links.loc[
        _get_intra_gb(network.links) & network.links.carrier.isin(["DC"]), "p_nom"
    ] = LARGE_NUMBER
    logger.info(
        "Set all intra-GB transmission line capacities and DC link capacities to very large values for copperplate model."
    )


def update_load_shedding(network: pypsa.Network, add_cost: float) -> None:
    """
    Modify the network in-place to remove the GB load shedding plants and set the
    marginal cost of all non-GB marginal plants to be at least as high as the
    most expensive marginal plant in Europe.

    Parameters
    ----------
    network : pypsa.Network
        The input PyPSA network representing the GB power system.
    add_cost : float
        The additional cost to add on top of the maximum European marginal plant cost.
    """
    # Limiting to just `PP` generators, which are the conventional, dispatchable plants.
    network.remove(
        "Generator",
        network.generators.filter(regex=r"GB \d+ Load Shedding", axis=0).index,
    )
    max_marginal_plant_cost = network.generators[
        (network.generators.set == "PP") & ~network.generators.bus.str.startswith("GB")
    ].marginal_cost.max()
    # add a cost to ensure they are always the most expensive
    network.generators.loc[
        network.generators.index.str.contains("Load Shedding"), "marginal_cost"
    ] = max_marginal_plant_cost + add_cost
    logger.info(
        "Removed GB load shedding generators and set Eur load shedding generator marginal costs to %.2f",
        max_marginal_plant_cost + add_cost,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, year=2022)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load network
    network = pypsa.Network(snakemake.input.network)

    copperplate_gb(network)
    if pd.notna(add_cost := snakemake.params["load_shedding_cost_above_marginal"]):
        update_load_shedding(network, add_cost)
    network.export_to_netcdf(snakemake.output.network)
