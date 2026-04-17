# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Additional GB electricity demand data processor.

This script creates a timeseries that reflects the missing electricity demand data from the FES workbook, which is not captured in the current modelled demand.
This includes direct transmission connections and T&D losses.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import get_scenario_name

logger = logging.getLogger(__name__)


def _read_timeseries_demand_data(filepath: str) -> pd.Series:
    """
    Read timeseries demand data from CSV file.

    Args:
        filepath (Path): Filepath to the CSV file containing timeseries demand data
    """
    df = (
        pd.read_csv(filepath, index_col=0)
        .stack()
        .rename_axis(index=["snapshot", "bus"])
        .rename("p_set")
        .filter(regex="GB ")
        .rename(Path(filepath).parent.stem)
    )
    return df


def _read_annual_demand_data(filepath: str, year: int) -> pd.Series | float:
    """
    Read annual demand data from CSV file.

    Args:
        filepath (Path): Filepath to the CSV file containing annual demand data
        year (int): Year for which to read demand data
    """
    df = (
        pd.read_csv(filepath)
        .query("year == @y", local_dict={"y": year})
        .drop(columns="year")
    )

    if "bus" in df.columns:
        df = df.set_index("bus").squeeze()
    else:
        df = df.squeeze().item()
    return df


def _load_and_process_fes_electricity_demand(
    filepath: str, year: int, scenario: str
) -> pd.Series:
    """
    Load and process total demand data from FES.

    Args:
        filepath (str): Filepath to the CSV file containing total demand data from FES
        year (int): Year for which to read demand data
        scenario (str): FES scenario to filter data for

    Returns:
        pd.Series: Series containing total demand from FES for the given year and scenario, indexed by Data item
    """
    df = pd.read_csv(filepath, parse_dates=["year"])
    df["year"] = df.year.dt.year
    df["Data item"] = df["Data item"].str.strip()

    df_filtered = df.query(
        "`year` == @y and `Pathway` == @scenario and `Peak/ Annual/ Minimum` == 'Annual [Fiscal]' and Unit == 'GWh'",
        local_dict={"y": year, "scenario": scenario},
    ).set_index("Data item")

    return df_filtered.data.mul(1e3)  # Convert from GWh to MWh


def get_additional_baseload(
    fes_total_demand: float, fes_losses: float, current_total: pd.Series
) -> pd.Series:
    """
    Get difference between current modelled electricity demand and total FES electricity demand.

    Demand will be applied as a static baseload per region, distributed relative to current annual demand per GB region.

    Args:
        fes_total_demand (float): Total electricity demand from FES for the given year and scenario
        fes_losses (float): T&D losses from FES for the given year and scenario
        current_total (pd.Series): Series containing current annual electricity demand per GB region, indexed by bus

    Returns:
        pd.Series: Series containing additional hourly baseload demand per GB region, indexed by bus
    """

    regional_share = current_total / current_total.sum()
    additional_baseload = (
        fes_total_demand - fes_losses
    ) * regional_share - current_total

    logger.info(
        "Missing electricity demand totalling %d MWh. Adding as additional baseload demand.",
        additional_baseload.sum(),
    )
    baseload_hourly = additional_baseload / 8760
    return baseload_hourly


def get_losses_profile(fes_losses: float, non_loss_profile: pd.Series) -> pd.Series:
    """
    Generate a profile of T&D losses based on current demand profile.

    Args:
        fes_losses (float): T&D losses from FES for the given year and scenario
        non_loss_profile (pd.Series): Series containing current annual electricity demand per GB region, indexed by bus and snapshots

    Returns:
        pd.Series: Series containing T&D losses per GB region and snapshot, indexed by bus and snapshots
    """
    normalised_profile = non_loss_profile / non_loss_profile.sum()
    losses_profile = fes_losses * normalised_profile
    logger.info("Adding T&D losses totalling %d MWh", losses_profile.sum())
    return losses_profile


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, data="fes_ev_dsm")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    year = int(snakemake.wildcards.year)
    scenario = get_scenario_name(snakemake)
    electricity_demands = pd.concat(
        [_read_timeseries_demand_data(f) for f in snakemake.input.electricity_demands],
        axis=1,
    )
    h2_demand = _read_annual_demand_data(snakemake.input.h2_demand, year)
    non_networked_h2_demand = _read_annual_demand_data(
        snakemake.input.non_networked_h2_demand, year
    )
    electrolysis_efficiency = _read_annual_demand_data(
        snakemake.input.electrolysis_efficiency, year
    )
    total_h2_electricity_demand = (
        h2_demand / electrolysis_efficiency + non_networked_h2_demand
    )
    current_total = electricity_demands.sum(axis=1) + (
        total_h2_electricity_demand / 8760
    )

    fes_electricity_demand = _load_and_process_fes_electricity_demand(
        snakemake.input.fes_total_electricity_demands, year, scenario
    )

    fes_total = fes_electricity_demand.loc[snakemake.params.total].sum()
    fes_losses = fes_electricity_demand.loc[snakemake.params.losses].sum()

    additional_baseload = get_additional_baseload(
        fes_total, fes_losses, current_total.groupby("bus").sum()
    )
    losses_profile = get_losses_profile(
        fes_losses,
        current_total + additional_baseload,
    )

    final_profile = additional_baseload + losses_profile
    assert np.isclose(
        model_total := final_profile.sum() + current_total.sum(), fes_total
    ), (
        "Final total demand does not match FES total demand after adding additional baseload and losses. "
        f"Expected {fes_total:.2f} MWh, obtained {model_total:.2f} MWh"
    )
    logger.info(
        "Final demand total: %d TWh, of which %d TWh is GSP-level demand & electrolysis, %d TWh is additional baseload and %d TWh is T&D losses",
        model_total / 1e6,
        current_total.sum() / 1e6,
        additional_baseload.sum() * 8760 / 1e6,
        losses_profile.sum() / 1e6,
    )
    final_profile.unstack("bus").to_csv(snakemake.output.csv)
