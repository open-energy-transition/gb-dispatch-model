# SPDX-FileCopyrightText: gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT

"""
Base electricity demand timeseries clustering.

This script clusters the regional base electricity demand by buses and saves it in CSV file format.
"""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def cluster_demand_timeseries(
    load_path: str,
    busmap_path: str,
    scaling: float = 1.0,
) -> pd.DataFrame:
    """
    Cluster the regional gb data to obtain required electricity demand timeseries by bus

    Args:
        load_path(str): Filepath to the regional data file containing the electricity demand timeseries
        busmap_path(str): Filepath to the CSV file containing the mapping between the clustered buses and buses in the base PyPSA network
        scaling(float) (optional): scaling factor to scale the demand timeseries

    Returns:
        pd.DataFrame : pandas dataframe containing the clustered electricity demand by bus indexed by the snapshots
    """

    # Read the electricity demand base .nc file
    load = (
        xr.open_dataarray(load_path)
        .to_dataframe()
        .squeeze(axis=1)
        .unstack(level="time")
    )

    # apply clustering busmap
    logger.info("Clustering the base electricity demand using busmap")
    busmap = pd.read_csv(busmap_path, dtype=str)
    busmap = busmap.set_index("Index").squeeze()

    missing_buses = list(set(load.index) - set(busmap.index))
    if len(missing_buses) > 0:
        logger.error(f"Busmap missing for buses: {missing_buses}.")

    load_clustered = load.groupby(busmap).sum().T

    logger.info(f"Load data scaled by factor {scaling}.")
    load_clustered *= scaling

    return load_clustered


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, clusters="clustered")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the files
    load_path = snakemake.input.load
    busmap_path = snakemake.input.busmap

    # Load scaling factor
    scaling = snakemake.params.scaling_factor

    load = cluster_demand_timeseries(load_path, busmap_path, scaling)

    # Save the regional base electricity demand profiles
    load.to_csv(snakemake.output.csv_file)
    logger.info(
        f"Base electricity demand dataframe saved to {snakemake.output.csv_file}"
    )
