# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Synthesise European H2 data and integrate with GB dataset.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def synthesise_eur_H2_data(
    demand_annual: pd.DataFrame,
    gb_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Synthesise European H2 demand data and integrate with GB dataset.

    Parameters
    ----------
    demand_annual: pd.DataFrame
        Annual demand data for GB and Europe
    gb_dataset: pd.DataFrame
        Dataset containing GB-only bus information

    Returns
    -------
    pd.DataFrame
        Combined annual demand data for GB and Europe
    """
    # Filter European demand for relevant load types and years
    normalised_dataset = gb_dataset.squeeze() / demand_annual.squeeze()

    dataset_inc_eur = (
        normalised_dataset.groupby(level="year", group_keys=False)
        .apply(lambda x: x.fillna(x.mean()))
        .mul(demand_annual)
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

    demand_annual = pd.read_csv(
        snakemake.input.h2_demand, index_col=["bus", "year"]
    ).squeeze()
    gb_regional_dataset = pd.read_csv(
        snakemake.input.gb_only_dataset, index_col=["bus", "year"]
    ).squeeze()
    all_regional_dataset = synthesise_eur_H2_data(demand_annual, gb_regional_dataset)
    all_regional_dataset.to_csv(snakemake.output.csv)
