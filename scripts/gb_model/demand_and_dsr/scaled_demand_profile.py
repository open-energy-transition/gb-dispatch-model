# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Scale baseline electricity load profiles using annual and peak demand data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def scale_shape_with_annual_data(
    demand_annual: pd.DataFrame,
    demand_shape: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """
    Process the baseline electricity demand data for a particular year using annual data and demand shape.

    Parameters
    ----------
    demand_annual: pd.DataFrame
        Annual baseline electricity demand data for each bus and year
    demand_shape: pd.DataFrame
        Demand shape for each demand type
    year:
        Year used in the modelling
    """

    normalized_demand_shape = _normalize(demand_shape)
    # Group demand data by year and bus and filter the data for required year
    demand_this_year = demand_annual.xs(year, level="year")

    # Filtering those buses that are present in both the dataframes
    if diff_bus := set(
        demand_this_year.index.get_level_values("bus")
    ).symmetric_difference(set(normalized_demand_shape.columns)):
        logger.warning(
            "The following buses are missing baseline electricity demand profile or annual demand data and will be ignored: %s",
            diff_bus,
        )

    # Scale the profile by the annual demand from FES
    load = normalized_demand_shape.drop(columns=diff_bus).mul(demand_this_year["p_set"])
    assert not load.isnull().values.any(), (
        "NaN values found in processed baseline electricity load data"
    )
    assert np.isclose(
        out_sum := load.sum().sum(), in_sum := demand_this_year["p_set"].sum()
    ), (
        f"Total energy mismatch after scaling baseline electricity load data - "
        f"Expected: {in_sum:.2f} MWh, Obtained: {out_sum:.2f} MWh"
    )
    return load


def _normalize(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series so that its sum equals 1."""
    normalized = series / series.sum()
    return normalized


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            Path(__file__).stem,
            year="2022",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    log_suffix = "-" + "_".join(snakemake.wildcards) if snakemake.wildcards else ""
    logger = logging.getLogger(Path(__file__).stem + log_suffix)

    gb_demand_annual = pd.read_csv(
        snakemake.input.gb_demand_annual, index_col=["bus", "year"]
    )
    eur_demand_annual = pd.read_csv(
        snakemake.input.eur_demand_annual, index_col=["load_type", "bus", "year"]
    ).xs("baseline_electricity", level="load_type")
    all_demand_annual = pd.concat([gb_demand_annual, eur_demand_annual])

    resistive_heater_demand_annual = (
        pd.read_csv(snakemake.input.resistive_heater_demand, index_col=0)
        .sum()
        .rename_axis(index="bus")
    )
    all_demand_annual_minus_resistive_heaters = all_demand_annual.sub(
        resistive_heater_demand_annual, axis=0
    )

    profile_shape = pd.read_csv(
        snakemake.input.demand_shape, index_col=[0], parse_dates=True
    )
    load_profile = scale_shape_with_annual_data(
        demand_annual=all_demand_annual_minus_resistive_heaters,
        demand_shape=profile_shape,
        year=int(snakemake.wildcards.year),
    )

    load_profile.to_csv(snakemake.output.csv)
