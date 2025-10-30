# SPDX-FileCopyrightText: gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT

"""
Hydrogen supply data processor.

This script processes hydrogen supply data from FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import pre_format

logger = logging.getLogger(__name__)


def parse_supply_data(
    supply_sheets: list,
    supply_sheets_mapping: dict,
    year_range: list,
    exogeneous_supply_list: list,
) -> pd.DataFrame:
    """
    Parse the hydrogen supply data to the required format.

    Args:
        supply_sheets (list): List of file paths to supply sheet CSV files
        supply_sheets_mapping (dict): Nested mapping of fes_year -> data type -> sheet_name
        year_range (list): Two-element list [start_year, end_year] defining the year range to include

    Returns:
        pd.DataFrame: Processed hydrogen supply data with MultiIndex ['bus', 'year'] and 'data' values in MWh.
    """
    # Retrieve annual supply data from sheets
    for year, sheets in supply_sheets_mapping.items():
        for data_type, sheet_name in sheets.items():
            # Find the corresponding file in the supply_sheets
            if data_type == "hydrogen_supply":
                sheet_file = next(f for f in supply_sheets if sheet_name in f)

    # Read the supply sheet data
    supply_data = pd.read_csv(sheet_file)

    # Pre-format the data: strip strings, standardize year, convert data to numeric
    supply_data = pre_format(supply_data)

    # Select exogeneous hydrogen supply data
    supply_data = supply_data[
        supply_data["carrier"].str.lower().isin(exogeneous_supply_list)
    ]

    # Filter by year range
    supply_data = supply_data[supply_data["year"].between(year_range[0], year_range[1])]

    # Use lowercase for carrier names
    supply_data["carrier"] = supply_data["carrier"].str.lower()

    # Pivot the data: year as index, carriers as columns, data as values
    df_hydrogen_supply = supply_data.pivot_table(
        index="year",
        columns="carrier",
        values="data",
        aggfunc="sum",  # Sum if there are duplicates
        fill_value=0,  # Fill missing combinations with 0
    )

    # Convert to MWh
    df_hydrogen_supply = df_hydrogen_supply * 1e6  # TWh to MWh

    return df_hydrogen_supply


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file path
    supply_sheets = snakemake.input.supply_sheets

    # Load all params
    fes_scenario = snakemake.params.scenario
    year_range = snakemake.params.year_range
    supply_sheets_mapping = snakemake.params.fes_supply_sheets
    exogeneous_supply_list = snakemake.params.exogeneous_supply_list

    # Parse hydrogen supply data
    df_hydrogen_supply = parse_supply_data(
        supply_sheets,
        supply_sheets_mapping,
        year_range,
        exogeneous_supply_list,
    )

    # Save the hydrogen supply data
    df_hydrogen_supply.to_csv(snakemake.output.hydrogen_supply)
    logger.info("Hydrogen supply data saved.")
