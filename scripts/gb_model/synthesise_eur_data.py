# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Add European demand data to the GB demand dataset.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def add_eur_demand(
    gb_demand_annual: pd.DataFrame,
    eur_demand_annual: pd.DataFrame,
    gb_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add European demand data to the GB demand dataset.

    Parameters
    ----------
    gb_demand_annual: pd.DataFrame
        Annual demand data for GB
    eur_demand_annual: pd.DataFrame
        Annual demand data for Europe
    gb_dataset: pd.DataFrame
        Dataset containing GB bus information

    Returns
    -------
    pd.DataFrame
        Combined annual demand data for GB and Europe
    """
    # Filter European demand for relevant load types and years
    all_demand_annual = pd.concat([gb_demand_annual, eur_demand_annual])
    normalised_dataset = gb_dataset.squeeze() / all_demand_annual.squeeze()

    dataset_inc_eur = (
        normalised_dataset.groupby(level="year", group_keys=False)
        .apply(lambda x: x.fillna(x.mean()))
        .mul(all_demand_annual)
    )
    if (is_na := dataset_inc_eur.isna()).any():
        logger.info(
            "The following buses/years have NaN values in the combined dataset and will be dropped: %s",
            is_na[is_na].index.tolist(),
        )
        dataset_inc_eur = dataset_inc_eur.dropna()

    return dataset_inc_eur.to_frame(gb_dataset.name)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            Path(__file__).stem,
            demand_type="baseline_electricity",
            year="2022",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    log_suffix = "-" + "_".join(snakemake.wildcards) if snakemake.wildcards else ""
    logger = logging.getLogger(Path(__file__).stem + log_suffix)

    gb_demand_annual = pd.read_csv(
        snakemake.input.gb_demand_annual, index_col=["bus", "year"]
    ).squeeze()
    eur_demand_annual = (
        pd.read_csv(
            snakemake.input.eur_demand_annual, index_col=["load_type", "bus", "year"]
        )
        .xs(snakemake.params.demand_type, level="load_type")
        .squeeze()
    )

    gb_regional_dataset = pd.read_csv(
        snakemake.input.gb_only_dataset, index_col=["bus", "year"]
    ).squeeze()
    all_regional_dataset = add_eur_demand(
        gb_demand_annual, eur_demand_annual, gb_regional_dataset
    )
    all_regional_dataset.to_csv(snakemake.output.csv)
