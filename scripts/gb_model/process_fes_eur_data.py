# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
European future capacity data table generator.

This is a script to clean up the European FES-compatible scenario data table.
"""

import logging
from pathlib import Path

import country_converter as coco
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def parse_inputs(
    df: pd.DataFrame,
    fes_scenario: str,
    year_range: list,
    countries: list[str],
) -> pd.DataFrame:
    """
    Parse the input data to the required format.

    Args:
        df (pd.DataFrame): FES-compatible European supply data table
        fes_scenario (str): FES scenario
        year_range (list): range of years to include
        countries (list[str]): list of countries to include
    """
    countries_set = set(countries) - {"GB"}
    country_codes = {x: coco.convert(x, to="ISO2") for x in df["Country"].unique()}
    df["bus"] = df["Country"].replace(country_codes)
    if any(df["bus"] == "not found"):
        logger.warning(
            f"Some countries could not be converted to ISO2: {df[df['bus'] == 'not found']['Country'].unique()}"
        )
    if any(countries_set.difference(df["bus"])):
        logger.error(
            f"Some European countries were not found in the dataset: {df[~df['bus'].isin(countries)]['Country'].unique()}"
        )

    df_pivoted = (
        df[df.bus.isin(countries_set)]
        .set_index([i for i in df.columns if not i.isnumeric()])
        .rename_axis(columns="year")
        .stack()
        .rename("data")
        .reset_index()
    )
    df_pivoted["year"] = df_pivoted["year"].astype(int)
    df_cleaned = df_pivoted[
        (df_pivoted["FES Scenario Alignment"] == fes_scenario)
        & (df_pivoted["year"].isin(range(year_range[0], year_range[1] + 1)))
    ]
    return df_cleaned


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file paths
    data_table = pd.read_csv(snakemake.input.eur_supply)

    # Load all the params
    fes_scenario = snakemake.params.scenario
    year_range = snakemake.params.year_range
    countries = snakemake.params.countries

    df = parse_inputs(data_table, fes_scenario, year_range, countries)
    logger.info(f"Extracted the {fes_scenario} relevant data")

    df.to_csv(snakemake.output.csv, index=False)
