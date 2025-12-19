# SPDX-FileCopyrightText: gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT

"""
Hydrogen storage data processor.

This script processes hydrogen storage data from FES workbook.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import pre_format

logger = logging.getLogger(__name__)


def parse_storage_data(
    storage_sheets: list,
    storage_sheets_mapping: dict,
    fes_scenario: str,
    year_range: list,
    interpolation_method: str = "linear",
) -> pd.DataFrame:
    """
    Parse the hydrogen storage data to the required format.

    Args:
        storage_sheets (list): List of file paths to storage sheet CSV files
        storage_sheets_mapping (dict): Nested mapping of fes_year -> data type -> sheet_name
        fes_scenario (str): FES scenario name to filter by (e.g., "Leading the Way")
        year_range (list): Two-element list [start_year, end_year] defining the year range to include
        interpolation_method (str): Method for interpolating between 5-year data points
                                   ("linear", "s_curve", or "step"). Default is "linear".

    Returns:
        pd.DataFrame: Processed hydrogen storage data with year as index and storage types as columns.
                     Data values are in MWh. Data is interpolated from
                     5-year intervals to annual values using the specified interpolation method.

    Processing steps:
        1. Load storage data from CSV files based on mapping configuration
        2. Filter data by specified FES scenario
        3. Standardize year format and convert data to numeric
        4. Apply interpolation to fill annual data between 5-year intervals
        5. Filter by specified year range
        6. Return processed DataFrame with annual storage capacity data e_nom in MWh
    """
    # Retrieve annual storage data from sheets
    for year, sheets in storage_sheets_mapping.items():
        for data_type, sheet_name in sheets.items():
            # Find the corresponding file in the storage_sheets
            if data_type == "storage_capacity":
                sheet_file = next(f for f in storage_sheets if sheet_name in f)

    # Read the storage sheet data
    storage_data = pd.read_csv(sheet_file)

    # Pre-format the data: strip strings, standardize year, convert data to numeric
    storage_data = pre_format(storage_data)

    # Select the scenario data
    storage_data = storage_data[storage_data["scenario"].str.lower() == fes_scenario]

    # Perform interpolation to get annual data
    storage_data_interpolated = interpolate_yearly_data(
        storage_data,
        method=interpolation_method,
        year_column="year",
        data_column="data",
    )

    # Filter by year range
    storage_data_interpolated = storage_data_interpolated[
        storage_data_interpolated["year"].between(year_range[0], year_range[1])
    ]

    # Prepare final dataframe
    storage_data_interpolated = storage_data_interpolated.set_index("year")["data"]
    storage_data_interpolated.name = "e_nom"

    # Convert to MWh
    storage_data_interpolated = storage_data_interpolated * 1e6  # TWh to MWh

    return storage_data_interpolated


def interpolate_yearly_data(
    df: pd.DataFrame,
    method: str,
    year_column: str,
    data_column: str,
) -> pd.DataFrame:
    """
    Interpolate data between 5-year intervals to get annual values.

    Args:
        df (pd.DataFrame): DataFrame with sparse year data (e.g., every 5 years)
        method (str): Interpolation method - "linear", "s_curve", or "step"
        year_column (str): Name of the year column
        data_column (str): Name of the data column to interpolate

    Returns:
        pd.DataFrame: DataFrame with interpolated annual data
    """

    # Create full year range
    min_year = df[year_column].min()
    max_year = df[year_column].max()
    full_years = np.arange(min_year, max_year + 1)

    # Get the sparse data points
    years = df[year_column].values
    values = df[data_column].values

    if method == "linear":
        # Linear interpolation
        f = interp1d(years, values, kind="linear", fill_value="extrapolate")
        interpolated_values = f(full_years)

    elif method == "s_curve":
        # S-curve (sigmoid-like) interpolation using cubic spline
        f = interp1d(years, values, kind="cubic", fill_value="extrapolate")
        interpolated_values = f(full_years)

        # Ensure non-negative values (optional)
        interpolated_values = np.maximum(interpolated_values, 0)

    elif method == "step":
        # Step interpolation - hold previous value until next data point
        f = interp1d(years, values, kind="previous", fill_value="extrapolate")
        interpolated_values = f(full_years)

    else:
        raise ValueError("Method must be 'linear', 's_curve', or 'step'")

    # Create new DataFrame with interpolated data
    result_df = pd.DataFrame(
        {year_column: full_years, data_column: interpolated_values}
    )

    # Add other columns if they exist (like scenario)
    for col in df.columns:
        if col not in [year_column, data_column]:
            # Use the first value for other columns (assuming they're constant)
            result_df[col] = df[col].iloc[0]

    return result_df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file path
    storage_sheets = snakemake.input.storage_sheet

    # Load all params
    fes_scenario = snakemake.params.scenario
    year_range = snakemake.params.year_range
    storage_sheets_mapping = snakemake.params.fes_storage_sheets
    interpolation_method = snakemake.params.interpolation_method

    # Parse storage data
    df_hydrogen_storage = parse_storage_data(
        storage_sheets,
        storage_sheets_mapping,
        fes_scenario,
        year_range,
        interpolation_method,
    )

    # Save the storage data
    df_hydrogen_storage.to_csv(snakemake.output.csv)
