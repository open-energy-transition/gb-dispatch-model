# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Regional GB data processor.

This script splits national data into regions from the FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import get_regional_distribution

logger = logging.getLogger(__name__)


def synthesise_regional_data(
    national_gb_data_path: str,
    regional_gb_data_path: str,
    regional_distribution_reference: str | list[str],
) -> pd.DataFrame:
    """
    Parse and regionally distribute national GB data using reference technology patterns.

    Args:
        national_gb_data_path (str):
            Filepath to the CSV file containing annual, national capacity data indexed by year
        regional_gb_data_path (str):
            Filepath to regional GB data CSV file for calculating regional distribution patterns
        regional_distribution_reference (str):
            Technology detail to use as reference for regional distribution patterns (e.g., EV charging infrastructure)

    Returns:
        pd.Series:
            Series with MultiIndex ['bus', 'year', ...] containing regionally distributed capacity in MW.
            Each region gets a proportional share of the annual capacity based on reference technology distribution.

    Processing steps:
        1. Load annual capacity data and regional reference technology data
        2. Calculate regional distribution patterns from reference technologies
        3. Apply regional distribution to annual capacity data
    """
    national_gb_data = pd.read_csv(national_gb_data_path)

    if len(national_gb_data.columns) > 2:
        national_gb_data = national_gb_data.pivot(
            index="year",
            values=national_gb_data.columns[-1],
            columns=national_gb_data.columns.drop("year")[:-1],
        )
    else:
        national_gb_data = national_gb_data.set_index("year")
    national_gb_data = national_gb_data.squeeze()

    # Load regional GB data
    regional_gb_data = pd.read_csv(regional_gb_data_path)

    # Obtain regional reference for distribution
    if isinstance(regional_distribution_reference, str):
        regional_distribution_reference = [regional_distribution_reference]
    regional_reference = regional_gb_data[
        regional_gb_data["Technology Detail"]
        .str.lower()
        .isin([x.lower() for x in regional_distribution_reference])
    ]

    # Group by bus and year
    regional_reference = regional_reference.groupby(["bus", "year"])["data"].sum()

    # Get regional distribution
    regional_dist = get_regional_distribution(regional_reference)

    # Fillna values with 0
    regional_dist = regional_dist.fillna(0)

    # Distribute data regionally
    regionalised = national_gb_data.multiply(regional_dist, axis="index")
    if isinstance(regionalised, pd.DataFrame):
        regionalised = regionalised.stack()

    return regionalised


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, data="fes_ev_dsm")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the inputs
    national_gb_data_path = snakemake.input.national_gb_data
    regional_gb_data_path = snakemake.input.regional_gb_data

    # Parse input data
    source_ref = snakemake.params.regional_distribution_reference["source"]

    df_regional = synthesise_regional_data(
        national_gb_data_path,
        regional_gb_data_path,
        source_ref,
    )
    df_regional.name = snakemake.params.regional_distribution_reference["name"]
    df_regional.to_csv(snakemake.output.csv)
