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
    df_eur: pd.DataFrame, df_gb: pd.DataFrame, df_totals_by_type: pd.DataFrame
) -> pd.DataFrame:
    """
    Parse the input data to the required format.

    Args:
        df_eur (pd.DataFrame):
            DataFrame containing European demand data.
        df_gb (pd.DataFrame):
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
        df_totals_by_type_rel.mul(df_gb)
        .groupby(["bus", "year"], group_keys=False)
        .apply(_normalise)
    )
    distributed_demand = df_share_all_years.mul(annual_demand).dropna()

    assert np.allclose(
        distributed_demand.groupby(["bus", "year"]).sum().loc[df_eur.bus.unique()],
        df_eur.set_index(["bus", "year"]).loc[df_eur.bus.unique()].data,
    ), "Distributed demands do not sum to total annual demands!"
    return distributed_demand.to_frame("p_set")


def add_extra_demands(
    df_distributed: pd.DataFrame, df_gb: pd.DataFrame, extra_demands: list[str]
) -> pd.DataFrame:
    """
    Add any extra demands to the distributed demands DataFrame.


    Args:
        df_distributed (pd.DataFrame):
            DataFrame containing distributed demands.

        df_gb (pd.DataFrame):
            DataFrame containing GB demand data for different load types and years.

        extra_demands (list[str]):
            List of file paths to CSV files containing extra demands to be added.

    Returns:
        pd.DataFrame:
            DataFrame containing the combined demands.
    """
    rel_demand = (
        df_distributed.p_set.groupby(["bus", "year"]).sum()
        / df_gb.groupby("year").sum()
    )
    for extra_demand in extra_demands:
        df_extra = _load_demand(extra_demand)
        df_demand = (
            rel_demand.mul(df_extra)
            .to_frame("p_set")
            .assign(load_type=df_extra.name)
            .set_index("load_type", append=True)
        )
        logger.info(
            "Synthesising extra demand '%s' for Europe, summing to %d MWh",
            df_extra.name,
            df_demand.p_set.sum(),
        )
        df_distributed = pd.concat(
            [df_distributed, df_demand.reorder_levels(df_distributed.index.names)]
        )
    return df_distributed


def _load_demand(file_path: str) -> pd.DataFrame:
    name = Path(file_path).stem.removeprefix("regional_").removesuffix("_demand_annual")
    df = pd.read_csv(file_path).groupby("year").p_set.sum()
    return df.rename(name)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    df_eur = pd.read_csv(snakemake.input.eur_data).query("Variable == 'Demand (TWh)'")
    demands = [_load_demand(file) for file in snakemake.input.electricity_demands]
    df_gb = pd.concat(demands, axis=1).rename_axis(columns="load_type").stack()
    df_totals = pd.read_csv(
        snakemake.input.energy_totals, index_col=["country", "year"]
    ).xs(snakemake.params.base_year, level="year")

    df_totals_by_type = pd.concat(
        [
            df_totals[cols].sum(axis=1)
            for cols in snakemake.params.totals_to_demands.values()
        ],
        keys=snakemake.params.totals_to_demands,
        names=["load_type", "bus"],
    )

    df_distributed = distribute_demands(df_eur, df_gb, df_totals_by_type)
    df_distributed_mwh = df_distributed * 1e6  # Convert from TWh to MWh
    df_distributed_and_extra_demand = add_extra_demands(
        df_distributed_mwh, df_gb, snakemake.input.extra_demands
    )
    df_distributed_and_extra_demand.to_csv(snakemake.output.csv)
