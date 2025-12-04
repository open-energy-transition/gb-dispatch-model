# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
Distribute European demands to different load types.

We assume that the GB demands for each load type are representative of the
relative distribution of demands across Europe, and use this to distribute
the total European demands into base electricity, heating, and transport
demands.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def _normalise(series: pd.Series) -> pd.Series:
    return series / series.sum()


def distribute_demands(
    df_eur: pd.DataFrame, df_demand: pd.DataFrame, df_totals_by_type: pd.DataFrame
) -> pd.DataFrame:
    """
    Parse the input data to the required format.

    Args:
        df_eur (pd.DataFrame):
            DataFrame containing European demand data.
        df_demand (pd.DataFrame):
            DataFrame containing GB demand data for different load types and years.
        df_totals_by_type (pd.DataFrame):
            DataFrame containing total demands by type for all European countries for a given historical reference year.
    """

    # The importance of each demand type is weighted by the relative importance between the types in GB.
    df_totals_by_type_rel = (
        df_totals_by_type.div(df_totals_by_type.xs("GB", level="bus"))
        .groupby("bus", group_keys=False)
        .apply(_normalise)
    )
    annual_demand = df_eur.set_index(["bus", "year"]).data
    df_share_all_years = (
        df_totals_by_type_rel.mul(df_demand)
        .groupby(["bus", "year"], group_keys=False)
        .apply(_normalise)
    )
    distributed_demand = df_share_all_years.mul(annual_demand).dropna()

    assert np.allclose(
        distributed_demand.groupby(["bus", "year"]).sum().loc[df_eur.bus.unique()],
        df_eur.set_index(["bus", "year"]).loc[df_eur.bus.unique()].data,
    ), "Distributed demands do not sum to total annual demands!"
    return distributed_demand.to_frame("p_set")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    df_eur = pd.read_csv(snakemake.input.eur_data).query("Variable == 'Demand (TWh)'")
    demands = {
        Path(file).stem.removesuffix("_demand_annual"): pd.read_csv(file)
        .groupby("year")
        .p_set.sum()
        for file in snakemake.input.demands
    }
    df_demand = pd.concat(
        demands.values(), keys=demands.keys(), names=["load_type", "year"]
    )
    df_totals = pd.read_csv(snakemake.input.energy_totals, index_col=[0, 1]).xs(
        snakemake.params.base_year, level="year"
    )

    df_totals_by_type = pd.concat(
        [
            df_totals[cols].sum(axis=1)
            for cols in snakemake.params.totals_to_demands.values()
        ],
        keys=snakemake.params.totals_to_demands,
        names=["load_type", "bus"],
    )

    df_distributed = distribute_demands(df_eur, df_demand, df_totals_by_type)
    df_distributed_mwh = df_distributed * 1e6  # Convert from TWh to MWh
    df_distributed_mwh.to_csv(snakemake.output.csv)
