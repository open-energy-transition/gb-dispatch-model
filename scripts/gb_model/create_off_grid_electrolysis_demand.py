# SPDX-FileCopyrightText: gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT

"""
Electricity demand processor for off-grid electrolysis.

This script processes electricity demand for off-grid electrolysis from FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import get_regional_distribution, pre_format

logger = logging.getLogger(__name__)


def parse_electricity_demand(
    supply_sheets: list,
    supply_sheets_mapping: dict,
    year_range: list,
    grid_electrolysis_capacities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse the electricity demand data of non-grid electrolysis.

    This function processes electricity demand data for industrial and commercial
    off-grid electrolysis and distributes it regionally based on grid-connected
    electrolysis capacity patterns.

    Args:
        supply_sheets (list): List of file paths to supply sheet CSV files
        supply_sheets_mapping (dict): Nested mapping of fes_year -> data_type -> sheet_name
        year_range (list): Two-element list [start_year, end_year] defining the year range to include
        grid_electrolysis_capacities (pd.DataFrame): Grid-connected electrolysis capacities
                                                    with MultiIndex ['bus', 'year'] for regional distribution

    Returns:
        pd.DataFrame: Processed electricity demand data with MultiIndex ['bus', 'year']
                     and electricity demand values in MWh. Data is regionally distributed
                     based on grid-connected electrolysis capacity patterns.

    Processing steps:
        1. Load electricity demand data from supply sheets
        2. Filter for "I&C off grid electrolysis demand" data items
        3. Standardize year format and convert data to numeric
        4. Filter by specified year range
        5. Convert units from GWh to MWh
        6. Calculate regional distribution based on grid electrolysis capacities
        7. Apply regional distribution to electricity demand data
    """
    # Retrieve annual electricity demand for non-grid electrolysis data
    for year, sheets in supply_sheets_mapping.items():
        for data_type, sheet_name in sheets.items():
            # Find the corresponding file in the supply_sheets
            if data_type == "electricity_demand":
                sheet_file = next(f for f in supply_sheets if sheet_name in f)

    # Read the electricity demand sheet data
    electricity_demand_data = pd.read_csv(sheet_file)

    # Pre-format the data: strip strings, standardize year, convert data to numeric
    electricity_demand_data = pre_format(electricity_demand_data)

    # Make columns lowercase
    electricity_demand_data.columns = electricity_demand_data.columns.str.lower()

    # Filter by year range
    electricity_demand_data = electricity_demand_data[
        electricity_demand_data["year"].between(year_range[0], year_range[1])
    ]

    # Select non-grid electrolysis electricity demand
    electricity_demand_data = electricity_demand_data[
        electricity_demand_data["data item"].str.lower()
        == "i&c off grid electrolysis demand (lw only)"
    ]

    # Prepare final dataframe
    df_electricity_demand = electricity_demand_data.set_index("year")["data"]

    # Convert to MWh
    df_electricity_demand = df_electricity_demand * 1e3  # GWh to MWh

    # Calculate regional distribution of grid-connected electrolysis capacities
    electrolysis_distribution = get_regional_distribution(grid_electrolysis_capacities)

    # Apply regional distribution to electricity demand data
    df_electricity_demand = df_electricity_demand * electrolysis_distribution

    # Rename series to 'p_set'
    df_electricity_demand.name = "p_set"

    return df_electricity_demand


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file path
    supply_sheets = snakemake.input.supply_sheets
    grid_electrolysis_sheet = snakemake.input.grid_electrolysis_capacities

    # Load all params
    fes_scenario = snakemake.params.scenario
    year_range = snakemake.params.year_range
    supply_sheets_mapping = snakemake.params.fes_supply_sheets

    # Load grid-connected electrolysis capacities
    grid_electrolysis_capacities = pd.read_csv(
        grid_electrolysis_sheet, index_col=["bus", "year"]
    ).squeeze()

    # Parse electricity demand data for non-grid electrolysis
    df_electricity_demand = parse_electricity_demand(
        supply_sheets,
        supply_sheets_mapping,
        year_range,
        grid_electrolysis_capacities,
    )

    # Save the electricity demand data
    df_electricity_demand.to_csv(snakemake.output.electricity_demand)
    logger.info("Electricity demand data for non-grid electrolysis saved.")
