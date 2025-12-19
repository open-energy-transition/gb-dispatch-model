# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Calculate future annual H2 demand in Europe.
"""

import logging
from pathlib import Path

import country_converter as coco
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def add_eur_demand(
    gb_demands: pd.DataFrame, eur_demands: pd.DataFrame, countries: list[str]
) -> pd.DataFrame:
    """
    Add European H2 demand to GB model demand dataframe.

    Parameters
    ----------
    gb_demands: pd.DataFrame
        GB H2 demand dataframe
    eur_demands: pd.DataFrame
        European H2 demand dataframe
    countries: list[str]
        List of European country codes to keep

    Returns
    -------
    pd.DataFrame
        Combined GB and European H2 demand dataframe
    """
    eur_demands.columns = eur_demands.columns.astype(int)
    eur_demands_all_years = (
        eur_demands.loc[countries]
        .drop("GB", errors="ignore")
        .T.reindex(sorted(gb_demands.index.get_level_values("year").unique()))
        .interpolate(method="slinear", fill_value="extrapolate", limit_direction="both")
        .clip(lower=0)
        .stack()
        .rename_axis(index=["year", "bus"])
        .to_frame("p_set")
    )
    all_df = pd.concat([gb_demands, eur_demands_all_years]).sort_index()
    return all_df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, data_type="unmanaged_charging")
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    gb_demand = pd.read_csv(snakemake.input.gb_demand, index_col=["year", "bus"])

    eur_demand_future = pd.read_csv(
        snakemake.input.eur_demand_tyndp, index_col="country"
    )
    eur_demand_today = pd.read_excel(
        snakemake.input.eur_demand_today,
        sheet_name="Annual consumption aggregated",
        skiprows=6,
        usecols="B:H",
    )
    # get annual demand in MWh per country
    eur_demand_today = (
        eur_demand_today.groupby("Country")["Clean consumption T/Y"]
        .sum()
        .mul(33.33 * 1000)
    )
    eur_demand_today.index = eur_demand_today.index.map(
        lambda x: coco.convert(x, to="ISO2")
    )
    eur_demand = pd.concat([eur_demand_today.to_frame(2022), eur_demand_future], axis=1)
    all_demand = add_eur_demand(gb_demand, eur_demand, snakemake.params.countries)

    all_demand.to_csv(snakemake.output.csv)
