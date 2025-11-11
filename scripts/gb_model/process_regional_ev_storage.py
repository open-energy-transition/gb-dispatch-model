# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Regional EV storage data processor.

This script prepares regional disaggregation of EV storage data.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import get_regional_distribution

logger = logging.getLogger(__name__)


def prepare_regional_ev_storage(
    storage_path: str, flexibility_path: str
) -> pd.DataFrame:
    """
    Prepare regional disaggregation of EV storage data using flexibility distribution patterns.

    Args:
        storage_path (str): Filepath to the EV storage data CSV file containing
                           annual storage capacity data indexed by year
        flexibility_path (str): Filepath to the flexibility data CSV file containing
                               regional flexibility data with MultiIndex [bus, year]

    Returns:
        pd.Series: Series with MultiIndex [bus, year] containing regionally distributed
                  EV storage capacity in GWh. Each region gets a proportional share of
                  the annual storage capacity based on flexibility distribution patterns.

    Processing steps:
        1. Load annual storage data and regional flexibility distribution
        2. Calculate regional distribution patterns from flexibility data
        3. Apply regional distribution to annual storage capacity
    """

    # Load storage data
    df_storage = pd.read_csv(storage_path, index_col=0)

    # Load flexibility data
    df_flexibility = pd.read_csv(flexibility_path, index_col=[0, 1])

    # Get regional distribution (1e-9 was added to avoid division by zero)
    regional_dist = get_regional_distribution(df_flexibility + 1e-9)

    # Disaggregate storage data regionally
    regional_storage = regional_dist["p_nom"] * df_storage["MWh"]

    # Rename series as MWh
    regional_storage = regional_storage.rename("MWh")

    return regional_storage


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the regional gb data file path
    storage_path = snakemake.input.storage
    flexibility_path = snakemake.input.flexibility

    # Prepare regional storage data
    df_regional_storage = prepare_regional_ev_storage(
        storage_path,
        flexibility_path,
    )

    # Write storage dataframe to csv file
    df_regional_storage.to_csv(snakemake.output.regional_storage)
    logger.info(
        f"Regional EV storage data saved to {snakemake.output.regional_storage}"
    )
