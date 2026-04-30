# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
Storage max-hours computation.

This is a script to compute the max_hours of storage plants from the storage volume and discharge capacity information in the FES workbook ES1 sheet.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import (
    get_scenario_name,
)

logger = logging.getLogger(__name__)


def get_max_hours(
    es1_path: str,
    tech_max_hours: list[str],
    year_range: list[int],
    fes_scenario: str,
    carrier_map: dict[str, str],
) -> pd.DataFrame:
    """
    Compute max hours for storage technologies based on FES ES1 workbook sheet

    Parameters
    ----------
    es1_path: str
        Path to ES1 sheet CSV file
    tech_max_hours: list[str]
        Technologies for which to compute the max hours
    year_range: list[int]
        Year range to filter
    fes_scenraio: str
        FES scenario to filter
    carrier_map: dict[str,str]
        Mapping of ES1 subtype to PyPSA carrier name
    """
    df_es1 = pd.read_csv(es1_path)
    df_es1_reqd = df_es1[
        (df_es1["Pathway"].str.lower() == fes_scenario.lower())
        & (
            df_es1["year"].between(
                year_range[0],
                year_range[1],
                inclusive="both",
            )
        )
    ]
    df_es1_grouped = (
        df_es1_reqd.query("SubType in @hours", local_dict={"hours": tech_max_hours})
        .groupby(["SubType", "year", "Variable"])
        .data.sum()
    )
    df_es1_unstacked = df_es1_grouped.unstack("Variable")
    df_max_hours = (
        df_es1_unstacked["Storage Capacity (GWh)"]
        / df_es1_unstacked["Capacity (MW)"]
        * 1000
    )
    df_max_hours.name = "max_hours"
    df_max_hours = pd.DataFrame(df_max_hours).reset_index()
    df_max_hours["SubType"] = df_max_hours["SubType"].map(
        lambda x: carrier_map[x] if x in carrier_map.keys() else x
    )

    return df_max_hours


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    fes_scenario = get_scenario_name(snakemake)

    df_max_hours = get_max_hours(
        es1_path=snakemake.input.es1_sheet,
        tech_max_hours=snakemake.params.tech_max_hours,
        year_range=snakemake.params.year_range,
        fes_scenario=fes_scenario,
        carrier_map=snakemake.params.tech_mapping,
    )

    df_max_hours.to_csv(snakemake.output.max_hours, index=False)
    logger.info(f"Exported max_hours information to {snakemake.output.max_hours}")
