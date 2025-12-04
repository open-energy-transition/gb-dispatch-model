# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Heating technology mix calculator.

This script extracts the heating technology mix from FES workbook and calculates the share of each technology in the final demand.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def _interpolation(x, year):
    """
    Interpolates the share of technology for the given year
    """
    num = (x["2050"] - x["2020"]) * (2050 - year)
    den = 2050 - 2020
    return x["2050"] - (num / den)


def process_fes_heatmix(
    fes_data_heatmix_path: str,
    electrified_heating_technologies: list[str],
    scenario: str,
    year: int,
) -> pd.DataFrame:
    """
    Process heating technology mix

    Args:
        fes_data_heatmix_path(str): Filepath to the heating mix CSV file for each sector
        electrified_heating_technologies (list[str]): List of technologies that contribute to electrified heating demand
        scenario (str): FES scenario considered for the modelling
        year (int): Modeling year
        busmap_path(str): Filepath to the CSV file containing the mapping between the clustered buses and buses in the base PyPSA network

    Returns:
        pd.DataFrame : pandas dataframe containing the share of heating technologies that contribute to electrified heating demand
    """

    # Read the FES data
    fes_data = pd.read_csv(
        fes_data_heatmix_path, index_col=["type", "2020", "scenario_2050"]
    )
    mask = fes_data.index.get_level_values("scenario_2050").str.contains(
        scenario, case=False
    )
    # Filter the data
    fes_data_filtered = (
        fes_data.loc[mask]
        .rename(columns={"data": "2050"})
        .reset_index("2020")
        .reset_index("scenario_2050", drop=True)
        .loc[electrified_heating_technologies]
        .astype(float)
    )

    # Compute the share for the required year by interpolating the shares between 2020 and 2050
    fes_data_filtered[year] = fes_data_filtered.apply(
        lambda x: _interpolation(x, year), axis=1
    )

    # Calculate % of homes supplied by each type of technology
    fes_data_filtered["share"] = fes_data_filtered[year] / fes_data_filtered[year].sum()

    return fes_data_filtered["share"]


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, year=2022)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    electrified_heating_technologies = snakemake.params.electrified_heating_technologies
    heating_mix = pd.DataFrame(
        index=electrified_heating_technologies, columns=["residential", "commercial"]
    )

    residential_share = process_fes_heatmix(
        fes_data_heatmix_path=snakemake.input.fes_residential_heatmix,
        electrified_heating_technologies=electrified_heating_technologies,
        scenario=snakemake.params.scenario,
        year=int(snakemake.params.year),
    )
    logger.info("Heating technology mix calculated for residential sector")

    commercial_share = process_fes_heatmix(
        fes_data_heatmix_path=snakemake.input.fes_commercial_heatmix,
        electrified_heating_technologies=electrified_heating_technologies,
        scenario=snakemake.params.scenario,
        year=int(snakemake.params.year),
    )
    logger.info("Heating technology mix calculated for commercial sector")

    heating_mix["residential"] = residential_share.tolist()
    heating_mix["commercial"] = commercial_share.tolist()
    heating_mix.index.name = "Technology"
    # Save the heating technology mix for residential and commercial sectors
    heating_mix.to_csv(snakemake.output.csv)
    logger.info(f"Heating technology mix saved to {snakemake.output.csv}")
