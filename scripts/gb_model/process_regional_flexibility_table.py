# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Regional Flexibility data processor.

This script splits flexibility data into regionsfrom the FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import get_regional_distribution

logger = logging.getLogger(__name__)


def parse_regional_flexibility_data(
    flexibility_data_path: str,
    regional_gb_data_path: str,
    regional_distribution_reference: list,
) -> pd.DataFrame:
    """
    Parse and regionally distribute flexibility data using reference technology patterns.

    Args:
        flexibility_data_path (str): Filepath to the flexibility data CSV file containing
                                   annual flexibility capacity data indexed by year
        regional_gb_data_path (str): Filepath to regional GB data CSV file for calculating
                                   regional distribution patterns
        regional_distribution_reference (list): List of technology details to use as reference
                                              for regional distribution patterns (e.g., EV charging infrastructure)

    Returns:
        pd.Series: Series with MultiIndex ['bus', 'year'] containing regionally distributed
                  flexibility capacity in MW. Each region gets a proportional share of the
                  annual flexibility capacity based on reference technology distribution.

    Processing steps:
        1. Load annual flexibility data and regional reference technology data
        2. Calculate regional distribution patterns from reference technologies
        3. Apply regional distribution to annual flexibility capacity data
    """
    # Load annual flexibility data
    flexibility_data = pd.read_csv(flexibility_data_path, index_col=0)

    # Load regional GB data
    regional_gb_data = pd.read_csv(regional_gb_data_path)

    # Obtain regional reference for distribution
    regional_reference = regional_gb_data[
        regional_gb_data["Technology Detail"]
        .str.lower()
        .isin(regional_distribution_reference)
    ]

    # Group by bus and year
    regional_reference = regional_reference.groupby(["bus", "year"])["data"].sum()

    # Get regional distribution
    regional_dist = get_regional_distribution(regional_reference)

    # Fillna values with 0
    regional_dist = regional_dist.fillna(0)

    # Distribute flexibility data regionally
    flexibility_regional = regional_dist * flexibility_data["p_nom"]

    # Set name as p_nom
    flexibility_regional.name = "p_nom"

    return flexibility_regional


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, flexibility_type="fes_ev_dsm")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the inputs
    flexibility_data_path = snakemake.input.flexibility
    regional_gb_data_path = snakemake.input.regional_gb_data

    # Parse input data
    flexibility_type = snakemake.wildcards.flexibility_type
    regional_distribution_reference = snakemake.params.regional_distribution_reference[
        flexibility_type
    ]

    df_regional_flexibility = parse_regional_flexibility_data(
        flexibility_data_path,
        regional_gb_data_path,
        regional_distribution_reference,
    )

    # Write regional flexibility dataframe to csv file
    df_regional_flexibility.to_csv(snakemake.output.regional_flexibility)
    logger.info(
        f"Regional flexibility data saved to {snakemake.output.regional_flexibility}"
    )
