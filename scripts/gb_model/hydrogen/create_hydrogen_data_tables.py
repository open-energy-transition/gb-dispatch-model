# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Hydrogen demand data processor.

This script processes exogeneous hydrogen demand data from FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import get_scenario_name

logger = logging.getLogger(__name__)


def parse_hydrogen_data(
    fes_data: pd.DataFrame,
    fes_scenario: str,
    year_range: list,
    data_selection: dict,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Parse the hydrogen demand data from FES workbook to obtain hydrogen demand in the required format.

    Args:
        fes_data (pd.DataFrame): FES-compatible whole system data table
        fes_scenario (str): FES scenario name to filter the data for
        year_range (list): Two-element list [start_year, end_year] defining the year range
    Returns:
        networked_h2_supply: annual H2 supply from networked electrolysis (TWh)
        non_networked_electrolysis_demand: annual electricity demand for non-networked electrolysis (TWh)
        electrolysis_efficiency: efficiency of electrolysis from FES data (TWh H2 / TWh electricity)
        storage_data: annual hydrogen storage capacity (TWh)

    """

    filtered_data = fes_data[
        (fes_data["Fuel Type"].str.lower() == "hydrogen")
        & (fes_data["Pathway"].str.lower() == fes_scenario.lower())
        & (fes_data["Units"] == "TWh")
        & fes_data["year"].between(year_range[0], year_range[1], inclusive="both")
    ]
    networked_h2_supply = _select_data(
        filtered_data, data_selection["networked_h2_supply"]
    )
    networked_electrolysis_demand = _select_data(
        filtered_data, data_selection["networked_electricity_demand"]
    )

    electrolysis_efficiency = networked_h2_supply / networked_electrolysis_demand
    storage_data = _select_data(filtered_data, data_selection["storage"])
    non_networked_electrolysis_demand = _select_data(
        filtered_data, data_selection["non_networked_electricity_demand"]
    )

    return (
        networked_h2_supply,
        non_networked_electrolysis_demand,
        electrolysis_efficiency,
        storage_data,
    )


def _select_data(fes_data: pd.DataFrame, data_filter: dict) -> pd.Series:
    """
    Get unique data items from the FES data.

    Args:
        fes_data (pd.DataFrame): FES-compatible whole system data table

    Returns:
        pd.Series: Series of unique data items
    """
    for key, vals in data_filter.items():
        if not isinstance(vals, list):
            vals = [vals]
        fes_data = fes_data[(fes_data[key].str.lower().isin(v.lower() for v in vals))]
    # * 1e6 to convert TWh to MWh
    fes_data_mwh = fes_data.groupby("year").data.sum().mul(1e6)
    return fes_data_mwh


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    fes_data = pd.read_csv(snakemake.input.whole_system_data)

    # Parse demand data
    (
        networked_h2_supply,
        non_networked_electrolysis_demand,
        electrolysis_efficiency,
        storage_data,
    ) = parse_hydrogen_data(
        fes_data,
        get_scenario_name(snakemake),
        snakemake.params.year_range,
        snakemake.params.data_selection,
    )

    networked_h2_supply.to_csv(snakemake.output.hydrogen_demand)
    non_networked_electrolysis_demand.to_csv(snakemake.output.electricity_demand)
    electrolysis_efficiency.to_csv(snakemake.output.electrolysis_efficiency)
    storage_data.to_csv(snakemake.output.storage)
