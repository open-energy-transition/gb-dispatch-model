# SPDX-FileCopyrightText:  gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
Low carbon contract strike price processor
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def process_etys_caps(
    df: pd.DataFrame, scenario: str, boundaries: list, year_range: list
) -> pd.DataFrame:
    """
    Process ETYS output chart data for the given scenario.

    Args:
        df (pd.DataFrame): ETYS chart data containing boundary capabilities
        scenario (str):
            Scenario name for which to process the data.
            If not available in `df`, the average across all scenarios will be used.
        boundaries (list): List of boundaries to consider for processing
        year_range (list): List of years to include in the output DataFrame
    Returns:
        pd.DataFrame: DataFrame containing boundary capabilities indexed by boundary name and future year
    """
    # ETYS 2024 adds F to boundary names for no obvious reason
    df["Boundary"] = df["Boundary"].str.strip("Ff")

    df_filtered = df[
        (df.Boundary.isin(boundaries)) & (df["Percentile/Transfer"] == "Capability")
    ].drop("Percentile/Transfer", axis=1)
    if scenario in df_filtered.Scenario.unique():
        df_scenario = df_filtered[df_filtered.Scenario == scenario].drop(
            "Scenario", axis=1
        )
    else:
        logger.warning(
            f"Scenario '{scenario}' not found in ETYS data. "
            "Using average capability across all scenarios instead."
        )
        df_scenario = (
            df_filtered.set_index(["Boundary", "Scenario"]).groupby("Boundary").mean()
        )

    df_scenario.columns = df_scenario.columns.astype(int)
    df_scenario = df_scenario.reindex(
        columns=range(year_range[0], year_range[1] + 1)
    ).ffill(axis=1)
    return (
        df_scenario.stack()
        .to_frame("capability_mw")
        .rename_axis(index=["boundary_name", "year"])
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file paths
    future_cap_df = pd.read_excel(
        snakemake.input.future_caps, sheet_name=snakemake.params.sheet_name
    )
    current_cap_df = pd.read_csv(snakemake.input.current_caps)
    boundaries = current_cap_df.boundary_name.unique()
    # use short-form scenario name as that's what they use in the chart data.
    fes_scenario = snakemake.wildcards.fes_scenario
    processed_future_cap_df = process_etys_caps(
        future_cap_df, fes_scenario, boundaries, snakemake.params.year_range
    )
    processed_future_cap_df.to_csv(snakemake.output.csv)
