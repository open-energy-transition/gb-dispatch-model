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


def prepare_demand_shape(
    demand_path: str,
) -> pd.DataFrame:
    """
    Parse and prepare different demand profiles.

    This function processes regional demand data and calculates
    normalized demand profiles (shapes) for use in EV demand modeling.

    Args:
        demand_path (str): Path to the CSV file containing regional
                                demand data with buses as columns
                                   and time periods as index.

    Returns:
        pd.DataFrame: Normalized demand profiles with the same structure
                     as input but with values representing demand shares/proportions.
                     Each column (region) sums to 1.0 across all time periods.

    Processing steps:
        1. Load regional demand data from CSV file
        2. Calculate normalized demand profiles by dividing each region's demand
           by its total annual demand
        3. Return demand shape profiles for use in demand modeling
    """
    # Load demand of PyPSA-Eur
    demand = pd.read_csv(demand_path, index_col=0)

    # Obtain transport demand shape
    demand_shape = demand / demand.sum()

    return demand_shape


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, clusters="clustered")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file path
    demand_path = snakemake.input.pypsa_eur_demand_timeseries

    # Prepare profile shape for transport demand
    demand_shape = prepare_demand_shape(
        demand_path,
    )

    # Save the transport demand profiles
    demand_shape.to_csv(snakemake.output.demand_shape)
    logger.info(
        f"Transport demand profile shapes saved to {snakemake.output.demand_shape}"
    )
