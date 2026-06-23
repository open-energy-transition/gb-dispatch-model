# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Grid-connected hydrogen electrolysis processor.

This script prepares regional grid-connected hydrogen electrolysis capacities from FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import get_regional_distribution

logger = logging.getLogger(__name__)


def parse_grid_electrolysis_data(
    regional_gb_data_path: str,
) -> pd.DataFrame:
    """
    Parse and process grid-connected hydrogen electrolysis supply data.

    This function processes regional hydrogen electrolysis capacity data, calculates
    regional distributions, and redistributes unmapped capacities based on existing
    regional patterns.

    Args:
        regional_gb_data_path (str): Path to the regional GB data CSV file containing
                                   hydrogen electrolysis capacity data by region and year.

    Returns:
        pd.DataFrame: Processed grid-connected electrolysis capacities with MultiIndex
                     ['bus', 'year'] and 'data' values in original units in MW.
                     Includes both originally mapped capacities and redistributed
                     unmapped capacities.

    Processing steps:
        1. Filter regional data for hydrogen electrolysis technology
        2. Group mapped data by bus (region) and year
        3. Calculate regional distribution proportions for each year
        4. Identify and aggregate unmapped capacities (NaN bus values)
        5. Redistribute unmapped capacities based on regional distribution patterns
        6. Combine mapped and redistributed capacities into final dataset
    """
    # Read regional GB data
    regional_gb_data = pd.read_csv(regional_gb_data_path)

    # Select regional hydrogen electrolysis data
    electrolysis_data = regional_gb_data[
        (regional_gb_data["Technology Detail"].str.lower() == "hydrogen electrolysis")
    ]

    # Calculate regional grid-connected electrolysis capacities
    regional_grid_electrolysis_capacities = electrolysis_data.groupby(["bus", "year"])[
        "data"
    ].sum()

    # Calculate regional distribution of grid-connected electrolysis capacities
    electrolysis_distribution = get_regional_distribution(
        regional_grid_electrolysis_capacities
    )

    # Map unmapped grid-connected electrolysis data
    unmapped_grid_electrolysis_capacities = (
        electrolysis_data[electrolysis_data["bus"].isnull()]
        .groupby("year")["data"]
        .sum()
    )
    unmapped_grid_electrolysis_capacities = (
        unmapped_grid_electrolysis_capacities * electrolysis_distribution
    )

    # Combine mapped and unmapped grid-connected electrolysis data
    grid_electrolysis_capacities = (
        regional_grid_electrolysis_capacities + unmapped_grid_electrolysis_capacities
    )

    return grid_electrolysis_capacities.to_frame("p_nom")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file path
    regional_gb_data_path = snakemake.input.regional_gb_data

    # Parse grid connected electrolysis data
    grid_electrolysis_capacities = parse_grid_electrolysis_data(
        regional_gb_data_path,
    )

    # Save the grid-connected electrolysis capacities
    grid_electrolysis_capacities.to_csv(snakemake.output.csv)
