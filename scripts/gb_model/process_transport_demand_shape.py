# SPDX-FileCopyrightText: gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT

"""
EV demand profile processor.

This script prepares regional EV demand profiles.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def prepare_transport_demand_shape(
    transport_demand_path: str,
) -> pd.DataFrame:
    """
    Parse and prepare transport demand profiles.

    This function processes regional transport demand data and calculates
    normalized demand profiles (shapes) for use in EV demand modeling.

    Args:
        transport_demand_path (str): Path to the CSV file containing regional
                                   transport demand data with buses as columns
                                   and time periods as index.

    Returns:
        pd.DataFrame: Normalized transport demand profiles with the same structure
                     as input but with values representing demand shares/proportions.
                     Each column (region) sums to 1.0 across all time periods.

    Processing steps:
        1. Load regional transport demand data from CSV file
        2. Calculate normalized demand profiles by dividing each region's demand
           by its total annual demand
        3. Return demand shape profiles for use in EV demand modeling
    """
    # Load transport demand of PyPSA-Eur
    transport_demand = pd.read_csv(transport_demand_path, index_col=0)

    # Obtain transport demand shape
    transport_demand_shape = transport_demand / transport_demand.sum()

    return transport_demand_shape


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, clusters="clustered")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file path
    transport_demand_path = snakemake.input.transport_demand

    # Prepare profile shape for transport demand
    transport_demand_shape = prepare_transport_demand_shape(
        transport_demand_path,
    )

    # Save the transport demand profiles
    transport_demand_shape.to_csv(snakemake.output.transport_demand_shape)
    logger.info(
        f"Transport demand profile shapes saved to {snakemake.output.transport_demand_shape}"
    )
