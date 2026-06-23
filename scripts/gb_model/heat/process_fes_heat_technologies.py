# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Heating technology mix calculator.

This script extracts the heating technology mix from FES workbook and calculates the share of each technology in the final demand.
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import get_scenario_name

logger = logging.getLogger(__name__)

SECTOR_MAPPING = {
    "residential": "residential",
    "commercial": "services",
    "industrial": "services",
}


def process_fes_heat_technologies(
    fes_heat_technology_data: str,
    electrified_heating_technologies: dict[str, list[str]],
    scenario: str,
    year_range: list[int],
) -> pd.DataFrame:
    """
    Process heating technologies from FES data to calculate the share of each technology in meeting heat demand.

    Args:
        fes_heat_technology_data (str): Filepath to the heating technology uptake CSV file for each sector
        electrified_heating_technologies (list[str]): List of technologies that contribute to electrified heating demand
        scenario (str): FES scenario considered for the modelling
        year (int): Modeling year

    Returns:
        pd.DataFrame : pandas dataframe containing the share of heating technologies that contribute to electrified heating demand
    """

    # Read the FES data
    fes_data = pd.read_csv(fes_heat_technology_data)

    # Filter the data
    filtered_data = fes_data[
        (fes_data.Units == "TWh")
        & (fes_data["Pathway"].str.lower() == scenario.lower())
        & (fes_data["Fuel"].str.startswith("electricity"))
        & (fes_data["year"].between(year_range[0], year_range[1], inclusive="both"))
    ]

    sectoral_data = (
        filtered_data.assign(
            sector=filtered_data["Sector/Aggregation Level"].map(SECTOR_MAPPING),
        )
        .groupby(["sector", "year", "Technology"])
        .data.sum()
    )
    grouped_data_dict: dict[str, pd.Series] = defaultdict(pd.Series)

    for current_tech, new_tech in electrified_heating_technologies.items():
        if current_tech not in sectoral_data.index.get_level_values("Technology"):
            logger.warning(
                f"Technology '{current_tech}' not found in the dataset. Skipping."
            )
            continue
        current_tech_data = sectoral_data.xs(current_tech, level="Technology")
        if isinstance(new_tech, str):
            grouped_data_dict[new_tech] = pd.concat(
                [grouped_data_dict[new_tech], current_tech_data], axis=1
            ).sum(axis=1)
        elif isinstance(new_tech, list):
            # If splitting into multiple technologies, divide the demand equally
            to_add = current_tech_data.divide(len(new_tech))
            for tech in new_tech:
                grouped_data_dict[tech] = pd.concat(
                    [grouped_data_dict[tech], to_add], axis=1
                ).sum(axis=1)

    final_data = (
        pd.DataFrame(grouped_data_dict)
        .rename_axis(index=["sector", "year"], columns="technology")
        .stack()
    )

    assert np.isclose(final_data.sum(), filtered_data.data.sum()), (
        "The total demand after grouping does not match the original demand."
    )

    return final_data.mul(1e6)  # TWh to MWh


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    heating_technology_data = process_fes_heat_technologies(
        fes_heat_technology_data=snakemake.input.fes_heat_technology_data,
        electrified_heating_technologies=snakemake.params.electrified_heating_technologies,
        scenario=get_scenario_name(snakemake),
        year_range=snakemake.params.year_range,
    )
    for sector in heating_technology_data.index.get_level_values("sector").unique():
        sector_data = heating_technology_data.xs(sector, level="sector")
        sector_data.to_csv(snakemake.output[sector])
