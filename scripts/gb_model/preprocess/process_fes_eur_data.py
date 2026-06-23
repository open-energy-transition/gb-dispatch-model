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
from scripts.gb_model._helpers import get_scenario_name

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
    df["bus"] = _eur_countries_to_buses(df["Country"])
    if any(df["bus"] == "not found"):
        logger.warning(
            f"Some countries could not be converted to ISO2: {df[df['bus'] == 'not found']['Country'].unique()}"
        )
    if any(countries_set.difference(df["bus"])):
        logger.error(
            f"Some European countries were not found in the dataset: {df[~df['bus'].isin(countries)]['Country'].unique()}"
        )

    df_filtered = df[
        df.bus.isin(countries_set)
        & (df["FES Pathway Alignment"].str.lower() == fes_scenario.lower())
        & (df["year"].between(year_range[0], year_range[1], inclusive="both"))
    ]

    return df_filtered


def _eur_countries_to_buses(countries: pd.Series) -> pd.Series:
    """
    Convert a list of European country codes to corresponding bus names.

    Parameters
    ----------
    countries: pd.Series
        European country codes

    Returns
    -------
    pd.Series
        List of corresponding bus names
    """
    # SEM = single electricity market (Ireland and Northern Ireland)
    countries_ie = countries.replace({"SEM": "Ireland"})
    country_codes = {x: coco.convert(x, to="ISO2") for x in countries_ie.unique()}
    countries_iso2 = countries_ie.replace(country_codes)
    return countries_iso2


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file paths
    data_table = pd.read_csv(snakemake.input.eur_data)

    # Load all the params
    fes_scenario = get_scenario_name(snakemake)
    year_range = snakemake.params.year_range
    countries = snakemake.params.countries

    df = parse_inputs(data_table, fes_scenario, year_range, countries)

    df.to_csv(snakemake.output.csv, index=False)
