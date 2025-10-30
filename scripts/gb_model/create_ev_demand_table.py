# SPDX-FileCopyrightText: gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT

"""
EV demand data processor.

This script processes EV demand data from FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def parse_ev_demand(
    regional_gb_data_path: str,
    fes_scenario: str,
    year_range: list,
) -> pd.DataFrame:
    """
    Parse the electric vehicle (EV) demand data to the required format.

    Args:
        regional_gb_data_path (str): Path to the regional GB data CSV file containing
                                   EV demand data by region and year
        fes_scenario (str): FES scenario name to filter by (e.g., "leading the way")
        year_range (list): Two-element list [start_year, end_year] defining the year range to include

    Returns:
        pd.DataFrame: Processed EV demand data with MultiIndex ['bus', 'year'] and
                     'data' values representing total EV electricity demand in original units.
                     Combines both "ev demand 1" and "ev demand 2" categories.

    Processing steps:
        1. Load regional GB data from CSV file
        2. Filter by specified FES scenario
        3. Filter by year range
        4. Select EV demand technologies ("ev demand 1" and "ev demand 2")
        5. Group by bus (region) and year, summing demand across EV categories
    """
    # Read regional GB data
    regional_gb_data = pd.read_csv(regional_gb_data_path)

    # Select FES scenario
    regional_gb_data = regional_gb_data[
        regional_gb_data["FES Scenario"].str.lower() == fes_scenario
    ]

    # Select year range
    regional_gb_data = regional_gb_data[
        regional_gb_data["year"].between(year_range[0], year_range[1])
    ]

    # Select EV demands
    ev_demand_data = regional_gb_data[
        (
            regional_gb_data["Technology Detail"]
            .str.lower()
            .isin(["ev demand 1", "ev demand 2"])
        )
    ]

    # Calculate regional EV demand
    regional_ev_demand = ev_demand_data.groupby(["bus", "year"])["data"].sum()

    # Convert from GWh to MWh
    regional_ev_demand = regional_ev_demand * 1000.0

    # Rename series to p_set
    regional_ev_demand.name = "p_set"

    return regional_ev_demand


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file path
    regional_gb_data_path = snakemake.input.regional_gb_data

    # Load all params
    fes_scenario = snakemake.params.scenario
    year_range = snakemake.params.year_range

    # Parse demand data
    df_ev_demand = parse_ev_demand(
        regional_gb_data_path,
        fes_scenario,
        year_range,
    )

    # Save the EV demand data
    df_ev_demand.to_csv(snakemake.output.ev_demand)
    logger.info(f"EV demand data saved to {snakemake.output.ev_demand}")
