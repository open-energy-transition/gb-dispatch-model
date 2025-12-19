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


def _get_intra_gb(df: pd.DataFrame) -> pd.Series:
    return df.bus0.str.startswith("GB") & df.bus1.str.startswith("GB")


def copperplate_gb(network: pypsa.Network) -> pypsa.Network:
    """
    Modify the network to represent a copperplate model of Great Britain by
    setting the capacities of all transmission lines to a very high value.

    Args:
        network (pypsa.Network): The input PyPSA network representing the GB power system.

    Returns:
        pypsa.Network: The modified PyPSA network with unconstrained transmission lines.
    """
    network.lines.loc[_get_intra_gb(network.lines), "s_nom"] = 1e6
    network.links.loc[
        _get_intra_gb(network.links) & network.links.carrier.isin(["DC"]), "p_nom"
    ] = 1e6
    logger.info(
        "Set all intra-GB transmission line capacities and DC link capacities to very large values for copperplate model."
    )
    return network


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, year=2022)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load network
    network = pypsa.Network(snakemake.input.network)

    network_copperplate = copperplate_gb(network)
    network_copperplate.export_to_netcdf(snakemake.output.network)
