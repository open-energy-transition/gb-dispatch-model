# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
EV V2G storage data processor.

This script processes required EV V2G storage data from the FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import parse_flexibility_data, pre_format

logger = logging.getLogger(__name__)


def parse_ev_v2g_storage_data(storage_sheet_path: str, scenario: str) -> pd.DataFrame:
    """
    Parse the EV V2G storage data from FES workbook to obtain storage capacity in the required format.

    Args:
        storage_sheet_path (str): Filepath to the storage data CSV file containing
                                 EV V2G storage capacity data by technology and year
        scenario (str): FES scenario name to filter the data for

    Returns:
        pd.DataFrame: DataFrame containing EV V2G storage capacity data indexed by year
                     with 'data' column representing storage capacity in GWh for
                     V2G (Vehicle-to-Grid) technology under Leading the Way scenario.

    Processing steps:
        1. Load storage data from CSV file with MultiIndex [technology, year]
        2. Filter for Leading the Way scenario (keep first occurrence of duplicates)
        3. Select V2G (Vehicle-to-Grid) storage technology
        4. Pre-format the dataframe for further processing
    """

    # Load storage data
    df_storage = pd.read_csv(storage_sheet_path, index_col=[0, 1, 2, 3])

    # Select EV V2G storage
    df_storage = df_storage.xs(
        (scenario, "V2G", "Storage Capacity (GWh)"),
        level=("scenario", "name", "parameter"),
    )

    # Pre-format dataframe
    df_storage = pre_format(df_storage.reset_index()).set_index("year").data

    return df_storage


def interpolate_storage_data(
    df_storage: pd.DataFrame,
    df_flexibility: pd.DataFrame,
    year_range: list[int],
) -> pd.DataFrame:
    """
    Interpolate the EV V2G storage data to match the required year range.

    Args:
        df_storage (pd.DataFrame): DataFrame containing EV V2G storage capacity data
                                    indexed by year.
        df_flexibility (pd.DataFrame): DataFrame containing flexibility data
                                       indexed by year.
        year_range (list[int]): List of years to interpolate the data for.

    Returns:
        pd.DataFrame: Interpolated DataFrame containing EV V2G storage data for the
                       specified year range.
    """
    # Compute energy to power ratio
    energy_to_power_ratio = df_storage / df_flexibility.loc[df_storage.index]
    logger.info(
        f"Computed energy to power ratio for EV V2G storage: {(-energy_to_power_ratio).to_dict()}"
    )

    # Compute mean energy to power ratio
    mean_energy_to_power_ratio = energy_to_power_ratio.mean()

    # Determine storage capacity based on flexibility data and energy to power ratio
    df_storage_interp = df_flexibility * mean_energy_to_power_ratio

    # Select only required years
    df_storage_interp = df_storage_interp.reset_index()
    df_storage_interp = df_storage_interp[
        df_storage_interp["year"].between(year_range[0], year_range[-1])
    ]

    # Convert GWh to MWh
    df_storage_interp["data"] = (df_storage_interp.data * 1000).abs()

    # Set name as MWh
    df_storage_interp = df_storage_interp.rename(columns={"data": "MWh"}).set_index(
        "year"
    )

    return df_storage_interp


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the regional gb data file path
    storage_sheet_path = snakemake.input.storage_sheet
    flexibility_sheet = pd.read_csv(snakemake.input.flexibility_sheet)

    # Load parameters
    fes_scenario = snakemake.params.scenario
    year_range = snakemake.params.year_range
    carrier_mapping = snakemake.params.carrier_mapping

    # Parse storage data
    df_storage = parse_ev_v2g_storage_data(storage_sheet_path, fes_scenario)

    # Parse flexibility data
    df_flexibility = parse_flexibility_data(
        flexibility_sheet, fes_scenario, year_range, carrier_mapping
    )

    # Interpolate storage data based on energy to power ratio
    df_storage_interp = interpolate_storage_data(
        df_storage,
        df_flexibility,
        year_range,
    )

    # Write storage dataframe to csv file
    df_storage_interp.to_csv(snakemake.output.storage_table)
    logger.info(f"EV V2G storage data saved to {snakemake.output.storage_table}")
