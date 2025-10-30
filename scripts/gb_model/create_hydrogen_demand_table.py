# SPDX-FileCopyrightText: gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT

"""
Hydrogen demand data processor.

This script processes exogeneous hydrogen demand data from FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import pre_format

logger = logging.getLogger(__name__)


def parse_demand_inputs(
    demand_sheets: list,
    demand_sheets_mapping: dict,
    fes_scenario: str,
    year_range: list,
) -> pd.DataFrame:
    """
    Parse the exogeneous hydrogen demand data to the required format.

    Args:
        demand_sheets (list): List of file paths to demand sheet CSV files
        demand_sheets_mapping (dict): Nested mapping of fes_year -> sector -> sheet_name
        fes_scenario (str): FES scenario name to filter by (e.g., "leading the way")
        year_range (list): Two-element list [start_year, end_year] defining the year range to include

    Returns:
        df_hydrogen_demand (pd.DataFrame): Processed hydrogen demand data with year as index
        and sectors as columns. Data values are in MWh for all sectors.
    """

    # Retrieve annual demand data from sheets
    demand_data = {}
    df_hydrogen_demand = pd.DataFrame()
    # Iterate through the nested structure: year -> sector -> sheet_name
    for year, sectors in demand_sheets_mapping.items():
        for sector, sheet_name in sectors.items():
            # Find the corresponding file in the demand_sheets
            # Look for files that contain both the year and sheet_name
            sheet_file = next(
                f for f in demand_sheets if sheet_name in f and str(year) in f
            )

            demand_data[sector] = pd.read_csv(sheet_file)

            # Pre-format the data: strip strings, standardize year, convert data to numeric
            demand_data[sector] = pre_format(demand_data[sector])

            # Make all column names lowercase
            demand_data[sector].columns = demand_data[sector].columns.str.lower()

            # Filter by year range
            demand_data[sector] = demand_data[sector][
                demand_data[sector]["year"].between(year_range[0], year_range[1])
            ]

            # Add scenario column for Road Transport
            if "scenario" not in demand_data[sector].columns:
                demand_data[sector]["scenario"] = fes_scenario

            # Select scenario
            demand_data[sector] = demand_data[sector][
                demand_data[sector]["scenario"].str.lower() == fes_scenario
            ]

            # Select hydrogen demand for road transport
            if sector == "road transport":
                demand_data[sector] = demand_data[sector][
                    demand_data[sector]["carrier"].str.lower() == "hydrogen"
                ]

            # Convert demand to MWh
            if sector in ["industrial", "commercial"]:
                demand_data[sector]["data"] = (
                    demand_data[sector]["data"] * 1e3
                )  # GWh to MWh
            else:
                demand_data[sector]["data"] = (
                    demand_data[sector]["data"] * 1e6
                )  # TWh to MWh

            # Define hydrogen demand dataframe
            if sector != "other":
                df_hydrogen_demand[sector] = demand_data[sector][
                    ["year", "data"]
                ].set_index("year")
            elif sector == "other":
                df_sector = demand_data[sector]

                for sub_sector in other_sectors_list:
                    df_hydrogen_demand[sub_sector] = (
                        df_sector[
                            (df_sector["fuel type"].str.lower() == "hydrogen")
                            & (df_sector["category lookup"].str.lower() == sub_sector)
                            & (df_sector["category"].str.lower() == "demand")
                        ]
                        .groupby("year")["data"]
                        .sum()
                    )

    return df_hydrogen_demand


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file path
    demand_sheets = snakemake.input.demand_sheets

    # Load all params
    fes_scenario = snakemake.params.scenario
    year_range = snakemake.params.year_range
    demand_sheets_mapping = snakemake.params.fes_demand_sheets
    other_sectors_list = snakemake.params.other_sectors_list

    # Parse demand data
    df_hydrogen_demand = parse_demand_inputs(
        demand_sheets,
        demand_sheets_mapping,
        fes_scenario,
        year_range,
    )

    # Save the hydrogen demand data
    df_hydrogen_demand.to_csv(snakemake.output.hydrogen_demand)
    logger.info(f"Hydrogen demand data saved to {snakemake.output.hydrogen_demand}")
