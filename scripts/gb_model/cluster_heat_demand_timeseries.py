# SPDX-FileCopyrightText: gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT

"""
heat demand timeseries clustering.

This script clusters the regional base electricity demand by buses and saves it in CSV file format.
"""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

def get_avg_cop_profiles(
    cop_path,
    clustered_population_layout_path,
    district_heat_share_path,
):
    clustered_population_layout=pd.read_csv(clustered_population_layout_path,index_col="name")

    district_heat_share=pd.read_csv(district_heat_share_path,index_col="country")
    #Current data exists from 1990-2021, Choosing 2021 to project data for all future years
    district_heat_share_2021=district_heat_share.loc[:,"2021"]

    clustered_population_layout['urban central'] = clustered_population_layout["ct"].map(district_heat_share_2021) * clustered_population_layout["urban"]
    clustered_population_layout['urban decentral'] = (1 - clustered_population_layout["ct"].map(district_heat_share_2021)) * clustered_population_layout["urban"]

    cop_profiles=(
        xr.open_dataset(cop_path)
        .to_dataframe()
        .squeeze(axis=1)
        .unstack(level=["heat_source","heat_system"])
    )
    cop_profiles.reset_index().set_index(["name","time"],inplace=True)

    heat_source=["air","ground"]
    heat_system=["urban central","urban decentral","rural"]

    cop_wt_avg=pd.DataFrame(index=cop_profiles.index)
    for source in heat_source:
        cop_profile_filter=cop_profiles[source]
        cop_wt_avg[source] = (cop_profile_filter * clustered_population_layout[heat_system]).sum(axis=1) 
        if source == "air":
            cop_wt_avg[source] /= clustered_population_layout["total"]
        else:
            cop_wt_avg[source] /= clustered_population_layout["rural"]

    return cop_wt_avg

def cluster_demand_timeseries(
    load_path: str,
    busmap_path: str,
) -> pd.DataFrame:
    """
    Cluster the regional gb data to obtain required electricity demand timeseries by bus

    Args:
        load_path(str): Filepath to the regional data file containing the electricity demand timeseries
        busmap_path(str): Filepath to the CSV file containing the mapping between the clustered buses and buses in the base PyPSA network

    Returns:
        pd.DataFrame : pandas dataframe containing the clustered electricity demand by bus indexed by the snapshots
    """
    breakpoint()

    # Read the electricity demand base .nc file
    load = (
        xr.open_dataset(load_path)
        .to_dataframe()
        .squeeze(axis=1)
        .unstack(level="snapshots")
    )
    # apply clustering busmap
    logger.info("Clustering the base electricity demand using busmap")
    busmap = pd.read_csv(busmap_path, dtype=str)
    busmap = busmap.set_index("Index").squeeze()

    missing_buses = list(set(load.index) - set(busmap.index))
    if len(missing_buses) > 0:
        logger.error(f"Busmap missing for buses: {missing_buses}.")

    load_clustered = load.groupby(busmap).sum().T

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
    cop_path=snakemake.input.cop_profile[0]
    clustered_population_layout=snakemake.input.clustered_pop_layout
    district_heat_share=snakemake.input.district_heat_share

    cop=get_avg_cop_profiles(cop_path, clustered_population_layout, district_heat_share)

    load = cluster_demand_timeseries(load_path, busmap_path)

    # Save the regional base electricity demand profiles
    load.to_csv(snakemake.output.csv_file)
    logger.info(
        f"Base electricity demand dataframe saved to {snakemake.output.csv_file}"
    )
