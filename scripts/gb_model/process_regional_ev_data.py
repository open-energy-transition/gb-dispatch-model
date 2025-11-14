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


def prepare_regional_ev_data(input_path: str, flexibility_path: str) -> pd.DataFrame:
    """
    Prepare regional disaggregation of EV data using flexibility distribution patterns.

    Args:
        input_path (str): Filepath to the EV data CSV file containing
                           annual aggregated data indexed by year
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

    # Load EV data
    df_ev = pd.read_csv(input_path, index_col=0)

    # Load flexibility data
    df_flexibility = pd.read_csv(flexibility_path, index_col=[0, 1])

    # Get regional distribution
    regional_dist = get_regional_distribution(df_flexibility)

    # Fillna values with 0
    regional_dist = regional_dist.fillna(0)

    # Disaggregate EV data regionally
    regional_ev = regional_dist["p_nom"] * df_ev.squeeze()

    # Keep original column name
    regional_ev = regional_ev.rename(df_ev.columns[0])

    return regional_ev


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, data_type="unmanaged_charging")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load input paths
    input_path = snakemake.input.input_csv
    flexibility_path = snakemake.input.flexibility

    # Prepare regional EV data
    df_regional_ev = prepare_regional_ev_data(
        input_path,
        flexibility_path,
    )

    # Write EV dataframe to csv file
    df_regional_ev.to_csv(snakemake.output.regional_output)
    logger.info(f"Regional EV data saved to {snakemake.output.regional_output}")
