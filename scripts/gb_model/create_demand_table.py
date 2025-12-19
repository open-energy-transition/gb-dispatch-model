# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Demand data processor.

This script processes required electricity demand data from the FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def parse_demand_data(
    regional_gb_data_path: str,
    technology_detail: list,
) -> pd.DataFrame:
    """
    Parse the regional gb data to obtain required electricity demand in the required format

    Args:
        regional_gb_data_path(str): Filepath to the regional data file containing the electricity demand
        technology_detail(list): List of technology details relevant to a particular demand type

    Returns:
        pd.DataFrame : MultiIndex dataframe containing the electricity demand indexed by PyPSA bus and year
    """
    df_regional_data = pd.read_csv(regional_gb_data_path)

    # Filter the data for required demand
    df_demand = df_regional_data.query(
        "`Technology Detail`.str.lower() in @technology_detail"
    )

    # Convert from GWh to MWh
    df_demand.loc[:, "data"] *= 1000

    df_demand_grouped = df_demand.groupby(["bus", "year"])["data"].sum()

    df_demand_grouped.name = "p_set"

    return df_demand_grouped


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the regional gb data file path
    regional_gb_data_path = snakemake.input.regional_gb_data

    # Parse input data
    demand_type = snakemake.wildcards.demand_type
    technology_detail = snakemake.params.technology_detail[demand_type]
    df_demand = parse_demand_data(regional_gb_data_path, technology_detail)

    # Write demand dataframe to csv file
    df_demand.to_csv(snakemake.output["demand"])
    logger.info(f"Electricity demand data saved to {snakemake.output['demand']}")
