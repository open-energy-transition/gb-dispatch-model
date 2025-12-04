# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Flexibility data processor.

This script processes required flexibility data from the FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import pre_format

logger = logging.getLogger(__name__)


def parse_flexibility_data(
    flexibility_data_path: str,
    technology_detail: list,
    fes_scenario: str,
    year_range: list,
) -> pd.DataFrame:
    """
    Parse the flexibility data to obtain required flexibility capacity in the required format.

    Args:
        flexibility_data_path (str): Filepath to the flexibility data CSV file containing
                                   flexibility capacity data by technology and year
        technology_detail (list): List of technology details relevant to a particular
                                flexibility type (e.g., EV DSM technologies)
        fes_scenario (str): FES scenario name to filter by (e.g., "leading the way")
        year_range (list): Two-element list [start_year, end_year] defining the year range to include

    Returns:
        pd.Series: Series containing aggregated flexibility capacity indexed by year.
                  Values represent total flexibility capacity in MW for the specified
                  technology types and scenario.

    Processing steps:
        1. Load and pre-format flexibility data from CSV file
        2. Filter by technology type, scenario, and year range
        3. Convert units from GW to MW and aggregate by year
    """
    df_flexibility_data = pd.read_csv(flexibility_data_path)

    # Pre-format: strip strings, standardize year, convert data to numeric
    df_flexibility_data = pre_format(df_flexibility_data)

    # Filter the data for required flexibility
    df_flexibility = df_flexibility_data.query(
        "`Detail`.str.lower() in @technology_detail"
    )

    # Select scenario
    df_flexibility = df_flexibility[
        df_flexibility["Scenario"].str.lower() == fes_scenario.lower()
    ]

    # Select year range
    df_flexibility = df_flexibility[
        df_flexibility["year"].between(year_range[0], year_range[1])
    ]

    # Convert from GW to MW
    df_flexibility.loc[:, "data"] *= 1000

    # Group by year
    df_flexibility_grouped = df_flexibility.groupby("year")["data"].sum()

    # Set series name
    df_flexibility_grouped.name = "p_nom"

    return df_flexibility_grouped


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, flexibility_type="fes_ev_dsm")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the regional gb data file path
    flexibility_data_path = snakemake.input.flexibility_sheet

    # Parse input data
    fes_scenario = snakemake.params.scenario
    year_range = snakemake.params.year_range
    flexibility_type = snakemake.wildcards.flexibility_type
    technology_detail = snakemake.params.technology_detail[flexibility_type]

    df_flexibility = parse_flexibility_data(
        flexibility_data_path,
        technology_detail,
        fes_scenario,
        year_range,
    )

    # Write flexibility dataframe to csv file
    df_flexibility.to_csv(snakemake.output["flexibility"])
    logger.info(f"Flexibility data saved to {snakemake.output['flexibility']}")
