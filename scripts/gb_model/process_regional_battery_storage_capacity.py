# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Regional battery storage data processor.

This script splits battery storage data into regions from the FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import parse_flexibility_data

logger = logging.getLogger(__name__)


def parse_regional_battery_storage_capacity(
    gb_e_nom: pd.DataFrame,
    regional_p_nom: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse and regionally distribute flexibility data using reference technology patterns.

    Args:
        gb_e_nom (pd.DataFrame): DataFrame containing annual storage capacity data indexed by year
        regional_p_nom (pd.DataFrame): DataFrame for calculating regional distribution patterns

    Returns:
        pd.Series: Series with MultiIndex ['bus', 'year'] containing regionally distributed
                  storage capacity in MW. Each region gets a proportional share of the
                  annual storage capacity based on reference technology distribution.

    """
    regional_battery_p_nom = (
        regional_p_nom[
            (regional_p_nom.carrier == "battery")
            & (regional_p_nom.bus.str.startswith("GB "))
        ]
        .set_index(["year", "bus"])
        .p_nom.groupby("year", group_keys=False)
        .apply(lambda x: x / x.sum())
    )

    regional_battery_storage = gb_e_nom * regional_battery_p_nom

    return regional_battery_storage.to_frame("e_nom")


def add_eur_battery_storage_capacity(
    regional_e_nom: pd.DataFrame,
    regional_p_nom: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add European battery storage capacity to the regional distribution.

    Args:
        regional_e_nom (pd.DataFrame): DataFrame containing annual storage capacity data indexed by year and distributed by region
        regional_p_nom (pd.DataFrame): DataFrame for calculating regional distribution patterns

    Returns:
        pd.Series: Series with MultiIndex ['bus', 'year'] containing updated regionally distributed
                  storage capacity in MW including European contribution.

    """
    regional_battery_p_nom = regional_p_nom[
        (regional_p_nom.carrier == "battery")
    ].set_index(["year", "bus"])

    ratio = (
        (regional_e_nom.e_nom / regional_battery_p_nom.p_nom)
        .groupby("year", group_keys=False)
        .apply(lambda x: x.fillna(x.mean()))
    )

    all_e_nom = ratio * regional_battery_p_nom.p_nom
    return all_e_nom.to_frame("e_nom")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the inputs
    flexibility_df = pd.read_csv(snakemake.input.flexibility_sheet)
    regional_gb_df = pd.read_csv(snakemake.input.regional_data)

    battery_capacity = parse_flexibility_data(
        flexibility_df,
        snakemake.params.scenario,
        snakemake.params.year_range,
        snakemake.params.carrier_mapping,
    )
    battery_capacity_mwh = battery_capacity * 1000  # Convert GWh to MWh
    # Parse input data
    df_regional = parse_regional_battery_storage_capacity(
        battery_capacity_mwh, regional_gb_df
    )
    df_regional_inc_eur = add_eur_battery_storage_capacity(df_regional, regional_gb_df)
    # Write regional flexibility dataframe to csv file
    df_regional_inc_eur.to_csv(snakemake.output.csv)
