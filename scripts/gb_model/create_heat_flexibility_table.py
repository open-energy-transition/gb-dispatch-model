# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Flexibility data processor.

This script processes required flexibility data from the FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def parse_heat_flexibility_data(
    flexibility_data: pd.DataFrame,
    fes_scenario: str,
    year_range: list[int],
    flex_level: str,
):
    """
    Parse and extract heat flexibility data from the FES flexibility dataset.

    Args:
        flexibility_data (pd.DataFrame): DataFrame containing flexibility data with MultiIndex
                                         ['year', 'flex_source', 'scenario']
        fes_scenario (str): FES scenario to filter data for (e.g., 'Steady Progression')
        year_range (list[int]): Slice of years to filter data for (e.g., [2020, 2030])
        flex_level (str): Flexibility level to use
    """
    # Data is stored as timestamps in upstream file, so we convert it here to integer year
    flexibility_data["year"] = flexibility_data["year"].dt.year
    flexibility_data_filtered = (
        flexibility_data.loc[
            (flexibility_data.scenario.str.lower() == fes_scenario)
            & (flexibility_data.year.between(*year_range, inclusive="both"))
        ]
        .drop("scenario", axis=1)
        .set_index(["year", "flex_source"])
    ).data

    peak = flexibility_data_filtered.xs(
        "Unconstrained peak electricity demand for heat", level="flex_source"
    )
    peak_with_flex = flexibility_data_filtered.xs(flex_level, level="flex_source")

    flex = peak - peak_with_flex
    assert (is_neg := flex >= 0).all(), (
        f"Flexibility values must be non-negative. Found: {flex[~is_neg]}"
    )
    return flex


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, flexibility_type="fes_ev_dsm")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the regional gb data file path
    flexibility_data = pd.read_csv(
        snakemake.input.flexibility_sheet,
        parse_dates=["year"],
    )

    # Parse input data
    fes_scenario = snakemake.params.scenario
    year_range = [int(i) for i in snakemake.params.year_range]
    flex_level = snakemake.params.flex_level

    df_flexibility = parse_heat_flexibility_data(
        flexibility_data, fes_scenario, year_range, flex_level
    )

    df_flexibility = (df_flexibility * 1000).to_frame("p_nom")  # Convert GW to MW

    # Write flexibility dataframe to csv file
    df_flexibility.to_csv(snakemake.output.csv)
