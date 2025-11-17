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

def process_cop_profiles(
    cop_path: str,
    clustered_population_layout_path: str,
    district_heat_share_path: str,
) -> pd.DataFrame :

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

    # To match the naming convention in FES data
    cop_wt_avg.rename(columns={'air':'ASHP','ground':'GSHP'},inplace=True)

    return cop_wt_avg

def process_fes_heatmix(
    fes_data_heatmix_path: str,
    scenario: str
) -> pd.DataFrame :

    electrified_demand_cols=["Electric resistive","Electric storage","ASHP","GSHP"]
    fes_data=pd.read_csv(fes_data_heatmix_path,index_col=[0,1,2])

    fes_data_filtered=fes_data[(fes_data
                                                .index.get_level_values(2)
                                                .str.lower().
                                                str.contains(scenario))]
    fes_data_filtered=fes_data_filtered.loc[electrified_demand_cols]
    fes_data_filtered['data']=fes_data_filtered['data'].apply(float)
    fes_data_filtered["share"]=fes_data_filtered["data"]/fes_data_filtered["data"].sum()       
    
    return fes_data_filtered

def cluster_demand_timeseries(
    load_path: str,
    fes_residential_share: pd.DataFrame,
    fes_commercial_share: pd.DataFrame,
    cop_profile: pd.DataFrame,
    output_path: str
) -> pd.DataFrame:
    """
    Cluster the regional gb data to obtain required electricity demand timeseries by bus

    Args:
        load_path(str): Filepath to the regional data file containing the electricity demand timeseries
        busmap_path(str): Filepath to the CSV file containing the mapping between the clustered buses and buses in the base PyPSA network

    Returns:
        pd.DataFrame : pandas dataframe containing the clustered electricity demand by bus indexed by the snapshots
    """

    # Read the electricity demand base .nc file
    load = (
        xr.open_dataset(load_path)
        .to_dataframe()
        .squeeze(axis=1)
        .unstack("node")
    )

    # Normalize the annual demand of each node
    load_normalized=load/load.sum()
    load_normalized=load_normalized.stack("node")

    scaled_load=pd.DataFrame(index=load_normalized.index)
    sectors=["residential","services"]
    system=["space","water"]

    for sector in sectors:
        for sys in system:
            load_reqd=load_normalized[f"{sector} {sys}"]
            if sector == 'residential':
                ashp_share=fes_residential_share.loc['ASHP','share'].values[0]
                gshp_share=fes_residential_share.loc['GSHP','share'].values[0]
            else:
                ashp_share=fes_commercial_share.loc['ASHP','share'].values[0]
                gshp_share=fes_commercial_share.loc['GSHP','share'].values[0]
            load_scaled=load_reqd*ashp_share/cop_profile['ASHP']
            load_scaled+= load_reqd*gshp_share/cop_profile['GSHP']
            scaled_load[f"{sector} {sys}"] = load_scaled

            # breakpoint()
    

    return scaled_load


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, clusters="clustered")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the files
    load_path = snakemake.input.load
    busmap_path = snakemake.input.busmap

    cop_profile=process_cop_profiles(
        cop_path=snakemake.input.cop_profile[0], 
        clustered_population_layout_path=snakemake.input.clustered_pop_layout, 
        district_heat_share_path=snakemake.input.district_heat_share
    )

    fes_residential_share=process_fes_heatmix(
        fes_data_heatmix_path=snakemake.input.fes_residential_heatmix,
        scenario=snakemake.params.scenario
    )

    fes_commercial_share=process_fes_heatmix(
        fes_data_heatmix_path=snakemake.input.fes_commercial_heatmix,
        scenario=snakemake.params.scenario
    )

    scaled_load = cluster_demand_timeseries(
        load_path=snakemake.input.load, 
        fes_residential_share=fes_residential_share,
        fes_commercial_share=fes_commercial_share,
        cop_profile=cop_profile,
        output_path=snakemake.output.csv_file)

    # Save the regional base electricity demand profiles
    scaled_load.to_csv(snakemake.output.csv_file)
    logger.info(
        f"Base electricity demand dataframe saved to {snakemake.output.csv_file}"
    )
